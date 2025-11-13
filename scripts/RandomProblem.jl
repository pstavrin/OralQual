using OralQual
using LinearAlgebra
using Random
using CairoMakie
using Distributions


@inline colmean(V::AbstractMatrix) = vec(mean(V, dims=2))

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

function randomLinearProblem(n::Int, d::Int, J::Int; rankdef=false, T=Float64, rng=Random.default_rng())
    # Random forward opertor
    H = rand(rng, T, n, d)
    L = (0.5)*rand(rng, T, n, n)
    Γ = Symmetric(L * L' + 1e-1I)


    if rankdef
        v = rand(d,1)
        v /= norm(v)
        H -= H*(v*v') # v ∈ ker(H)
        w = rand(n,1)
        w /= norm(w)
        H -= (w*w')*H # w ∈ ker(Hᵀ)
        basisH,~,~ = svd(H',full=true)
        basisV = hcat(v, basisH[:,2:d-2])
        V₀ = basisV*rand(d-2,J)
    else
        V₀ = rand(rng, T, d, J)
    end

    return Matrix{T}(H), Matrix{T}(Γ), Matrix{T}(V₀)
end

## Setup
n = 5
d = 8
J = 100000
H, Γ, V₀ = randomLinearProblem(n, d, J, rankdef=true)
truth = rand(d, 1)
ε = rand(MvNormal(Γ)) # noise
y = H*truth + ε # noisy data
y = vec(y)
H⁺ = pinv(H'*(Γ\H))*((H')/Γ) # weighted pseudoinverse
v_star = H⁺ * y
Hess = H'*(Γ\H)
pHessian = pinv(Hess)

## EKRMLE
obj = EKRMLEObj(V₀, y, Γ) # create EKRMLE object
iters = 100
H_s(::Nothing, v::AbstractVector) = H * v
EKRMLE_run!(obj, nothing, H_s, iters)

##
v_star - colmean(obj.V[end])
C = _samplecov(obj.V[end])
opnorm(pHessian - C)

## 
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1])
ax1.yreversed=true
ax2 = Axis(fig[1, 3])
ax2.yreversed=true
hm1 = heatmap!(ax1, C; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, pHessian; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)

## Spectral projectors
function spectralproj(H::AbstractArray{T},v₀::AbstractArray{T},Σ::AbstractArray{T}) where {T<:AbstractFloat}
    n,d = size(H)
    Γ = _samplecov(v₀)
    HGH = H*Γ*H'
    Hessian = H'*(Σ\H)
    H⁺ = pinv(Hessian)*(H'/Σ)
    # Observation space
    Λ,W = eigen(HGH,Σ) # solve GEV
    Λ = Λ[end:-1:1] # sort in descending order
    W = W[:,end:-1:1] # sort in descending order
    r = sum(broadcast(abs,Λ) .> 1e-10) # number of nonzero λ
    h = rank(H) # rank of H

    # normalize first r eigenvectors
    for i = 1 : r
        W[:,i] /= sqrt(W[:,i]'Σ*W[:,i])
    end

    basisSH = qr(Σ\H).Q # basis of range inv(Σ)H
    basiskerHT = nullspace(H') # basis of ker(Hᵀ)

    v = rand(n,n-r)
    for ℓ = r+1 : n
        if ℓ <= h
            w = basisSH*basisSH'v[:,ℓ-r]
        else
            w = basiskerHT*basiskerHT'*v[:,ℓ-r]
        end
        for k = 1:ℓ-1
            w = w - ((w'Σ*W[:,k])/(W[:,k]'Σ*W[:,k]))*W[:,k]
        end
        W[:,ℓ] = w/sqrt(w'Σ*w)
    end

    𝒫 = Σ*W[:,1:r]*W[:,1:r]'
    if h > r
        𝒬 = Σ*W[:,r+1:h]*W[:,r+1:h]'
    else
        𝒬 = zeros(n,n)
    end
    𝒮 = I-𝒫-𝒬

    # State space
    Λ,U = eigen(Γ*Hessian) # solve GEV
    Λ = Λ[end:-1:1] # sort in descending order
    U = U[:,end:-1:1] # sort in descending order
    # normalize first r eigenvectors
    for i = 1 : r
        U[:,i] /= sqrt(U[:,i]'Hessian*U[:,i])
    end
    #v = rand(n,n-r)
    for ℓ = r+1 : h
            w = basisSH*basisSH'v[:,ℓ-r]
            u = H⁺*Σ*w
        for k = 1:ℓ-1
            u = u - ((u'Hessian*U[:,k])/(U[:,k]'Hessian*U[:,k]))*U[:,k]
        end
        U[:,ℓ] = u/sqrt(u'Hessian*u)
    end

    ℙ = U[:,1:r]*U[:,1:r]'Hessian
    if h > r
        ℚ = U[:,r+1:h]*U[:,r+1:h]'Hessian
    else
        ℚ = zeros(d,d)
    end
    𝕊 = I-ℙ-ℚ

    return 𝒫,𝒬,𝒮,ℙ,ℚ,𝕊
end

~,~,~,P = spectralproj(H, V₀, Γ)
P = real.(P)

## 
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1])
ax1.yreversed=true
ax2 = Axis(fig[1, 3])
ax2.yreversed=true
hm1 = heatmap!(ax1, C; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, P*pHessian*P'; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)
