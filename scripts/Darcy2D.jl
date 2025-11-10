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
    φ::Matrix{T}
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
    τ::T

) where {T<:AbstractFloat,TI<:Int}

    X = Vector(LinRange(0,L,Nₓ))
    y_locs = Vector(Δ_obs_Y:Δ_obs_Y:Nₓ-Δ_obs_Y)
    x_locs = Vector(Δ_obs_X:Δ_obs_X:Nₓ-Δ_obs_X)
    n = length(y_locs)
    Δx = X[2]-X[1]
    f = eff_2D(X)
    logK,φ,λ,θ_true = KL_expansion(X,N_KL,α,τ)

    Darcy_params_2D(Nₓ,L,Δx,X,f,α,τ,N_KL,logK,φ,λ,θ_true,y_locs,x_locs,n,d)
    
end

function eff_2D(y::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(y)
    f = zeros(T, N, N)
    for i = 1:N
        if y[i] <= 4/6
            f[:, i] = 1000.0
        elseif (y[i] > 4/6 && y[i] <= 5/6)
            f[:, i] = 2000.0
        elseif y[i] > 5/6
            f[:, i] = 3000.0
        end

    end
    return f
    
end
