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

function randomLinearProblem(n::Int, d::Int, J::Int; T=Float64, rng=Random.default_rng())
    # Random forward opertor
    H = rand(rng, T, n, d)
    L = (0.5)*rand(rng, T, n)
    Γ = Symmetric(L * L' + 1e1*I)
    V₀ = rand(rng, T, d, J)

    return Matrix{T}(H), Matrix{T}(Γ), Matrix{T}(V₀)
end

## Setup
n = 50
d = 20
J = 100000
H, Γ, V₀ = randomLinearProblem(n, d, J)
truth = rand(d, 1)
ε = rand(MvNormal(Γ)) # noise
y = H*truth + ε # noisy data
y = vec(y)
H⁺ = pinv(H'*(Γ\H))*((H')/Γ) # weighted pseudoinverse
v_star = H⁺ * y
pHessian = pinv(H'*(Γ\H))

## EKRMLE
params = (; H)
obj = EKRMLEObj(V₀, y, Γ) # create EKRMLE object
iters = 50
H_single(params, v::AbstractVector) = params.H * v
EKRMLE_run!(obj, params, H_single, iters)

##
v_star - colmean(obj.V[end])
C = _samplecov(obj.V[end])
opnorm(pHessian - C)



