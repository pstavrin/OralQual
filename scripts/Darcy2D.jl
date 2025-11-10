using OralQual
using LinearAlgebra
using Random
using CairoMakie
using Distribution

## Helpers 

mutable struct Darcy_params_2D{T<:AbstractFloat, TI<:Int}
    # Discretization
    Nₓ::TI # mesh size
    L::T # domain
    Δx::T # discretization level
    X::Vector{T} # mesh

    # source term
    f::Matrix{T}

    # Covariance parameters
    α::T
    τ::T

    # KL parameters
    N_KL::TI
    logk_2d::Matrix{T}
    φ::Array{T, 3}
    λ::Vector{T}
    θ_true::Vector{T}

    # inversion
    y_locs::Vector{TI}
    x_locs::Vector{TI}
    n::TI
    d::TI

end

function Darcy_params_2D(
    Nₓ::TI,
    L::T,
    N_KL::TI,
    Δ_obs_X::TI,
    Δ_obs_Y::TI,
    d::TI,
    α::T,
    τ::T,
    σ₀::T=1.0

) where {T<:AbstractFloat,TI<:Int}

    X = Vector(LinRange(0,L,Nₓ))
    y_locs = Vector(Δ_obs_Y:Δ_obs_Y:Nₓ-Δ_obs_Y)
    x_locs = Vector(Δ_obs_X:Δ_obs_X:Nₓ-Δ_obs_X)
    n = length(y_locs)
    Δx = X[2]-X[1]
    f = eff_2D(X)
    logK,φ,λ,θ_true = KL_expansion_2D(X,N_KL,α,τ,σ₀)

    Darcy_params_2D(Nₓ,L,Δx,X,f,α,τ,N_KL,logK,φ,λ,θ_true,y_locs,x_locs,n,d)
    
end

function eff_2D(y::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(y)
    f = zeros(T, N, N)
    for i = 1:N
        if y[i] <= 4/6
            f[:, i] .= 3000.0
        elseif (y[i] > 4/6 && y[i] <= 5/6)
            f[:, i] .= 4000.0
        elseif y[i] > 5/6
            f[:, i] .= 5000.0
        end

    end
    return f
    
end



function get_pairs(N_KL::TI) where {TI<:Int}
    m = floor(Int, sqrt(2*N_KL)) + 1
    N_pairs = (m + 1)^2 - 1

    pairs = Array{Int}(undef, N_pairs, 2)
    mags = Array{Int}(undef, N_pairs)

    k = 0
    for i = 0:m, j = 0:m
        if i == 0 && j == 0
            continue
        end
        k += 1
        pairs[k, 1] = i
        pairs[k, 2] = j
        mags[k] = i^2 + j^2

    end

    pairs = pairs[sortperm(mags), :]
    return pairs[1:N_KL, :]
    
end


function KL_expansion_2D(x::Array{flt,1}, Nₖ::int, α::flt=2.0, τ::flt=3.0, σ₀::flt=1.0; rng::AbstractRNG=Random.default_rng()) where {flt<:AbstractFloat, int<:Int}
    N = length(x)
    X, Y = repeat(x, 1, N), repeat(x, 1, N)'
    pairs = get_pairs(Nₖ)

    φ = zeros(flt, N, N, Nₖ)
    λ = zeros(flt, Nₖ)

    for i = 1:Nₖ
        if (pairs[i, 1] == 0 && pairs[i, 2] == 0)
            φ[:, :, i] .= 1
        elseif pairs[i, 1] == 0
            φ[:, :, i] = sqrt(2)*cos.(pi * (pairs[i, 2]*Y))
        elseif pairs[i, 2] == 0
            φ[:, :, i] = sqrt(2)*cos.(pi * (pairs[i, 1]*X))
        else
            φ[:, :, i] = 2*cos(pi * (pairs[i, 2]*Y)) .* cos(pi * (pairs[i, 1]*X))
        end
        
        λ[i] = (pi^2 * (pairs[i, 1]^2 + pairs[i, 2]^2) + τ^2)^(-α)

    end

    𝛉 = rand(rng, Normal(0, σ₀), Nₖ)

    logk_2D = zeros(flt, N, N)
    for i = 1:Nₖ
        logk_2D .+= 𝛉[i]*sqrt(λ[i])*φ[:, :, i]
    end

    return logk_2D, φ, λ, 𝛉

    
end




## Setup
N, L = 80, 1.0
obs_ΔN = 10
α = 2.0
τ = 3.0
N_KL = 32
σ₀ = 1.0
d = N_KL
darcy = Darcy_params_2D(N, L, N_KL, obs_ΔN, obs_ΔN, d, α, τ, σ₀)
