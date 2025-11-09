export Darcy_params, eff, fwd, fwd_RLS, KL_expansion, get_logk, Darcy_1D_solver

mutable struct Darcy_params{T<:AbstractFloat, TI<:Int}
    # Discretization
    Nₓ::TI # mesh size
    L::T # domain
    Δx::T # discretization level
    X::Vector{T} # mesh

    # source term
    f::Vector{T}

    # Covariance parameters
    α::T
    τ::T

    # KL parameters
    N_KL::TI
    logk::Vector{T}
    φ::Matrix{T}
    λ::Vector{T}
    θ_true::Vector{T}

    # inversion
    y_locs::Vector{TI}
    n::TI
    d::TI

end


function Darcy_params(
    Nₓ::TI,
    L::T,
    N_KL::TI,
    Δ_obs::TI,
    d::TI,
    α::T,
    τ::T

) where {T<:AbstractFloat,TI<:Int}

    X = Vector(LinRange(0,L,Nₓ))
    y_locs = Vector(Δ_obs:Δ_obs:Nₓ-Δ_obs)
    n = length(y_locs)
    Δx = X[2]-X[1]
    f = eff(X)
    logK,φ,λ,θ_true = KL_expansion(X,N_KL,α,τ)

    Darcy_params(Nₓ,L,Δx,X,f,α,τ,N_KL,logK,φ,λ,θ_true,y_locs,n,d)
    
end

function eff(x::Array{flt,1}) where {flt<:AbstractFloat}
    # constructs f(x)
    N = length(x)
    f = zeros(flt, N)
    for i = 1 : N
        if x[i] <= 1/2
            f[i] = 5000.0
        else
            f[i] = 6000.0
        end
    end
    return f   
end

function KL_expansion(x::Array{flt,1}, Nₖ::int, α::flt=2.0, τ::flt=3.0; rng::AbstractRNG=Random.default_rng()) where {flt<:AbstractFloat, int<:Int}
    N = length(x)
    φ = zeros(flt, N, Nₖ)
    λ = zeros(flt, Nₖ)
    𝛉 = rand(rng, Normal(0,1), Nₖ)
    logK = zeros(flt, N)


    

    for ℓ = 1 : Nₖ
        φ[:,ℓ] = sqrt(2)*cos.(pi*ℓ .* x)
        λ[ℓ] = (pi^2*ℓ^2 + τ^2)^(-α)
        logK .+= 𝛉[ℓ]*sqrt(λ[ℓ])*φ[:,ℓ]
    end

    return logK, φ, λ, 𝛉

end


function Darcy_1D_solver(darcy::Darcy_params, κ::Array{flt,1}) where {flt<:AbstractFloat}
    Δx = darcy.Δx
    N = darcy.Nₓ
    f = darcy.f


    Δx2 = Δx^2
    C = 2*Δx2
    d = zeros(flt, N-2)
    du = zeros(flt, N-3)
    dd = zeros(flt, N-3)

    for i = 2:N-1
        d[i-1] = (κ[i-1] + 2*κ[i] + κ[i+1])/C
    end

    for i = 2:N-2
        du[i-1] = -(κ[i]+κ[i+1])/C
    end

    A = Tridiagonal(du,d,du) # symmetric?
    p = A\(f[2:N-1])[:]
    sol = zeros(flt,N)
    sol[2:N-1] .= p

    return sol

end


function get_logk(darcy::Darcy_params, v::AbstractVector{flt}) where {flt<:AbstractFloat}
    N = darcy.Nₓ
    Nₖ = darcy.N_KL
    φ = darcy.φ
    λ = darcy.λ
    d = length(v)
    @assert(d <= Nₖ)
    logk = zeros(flt,N)

    for ℓ = 1 : d
        logk .+= v[ℓ]*sqrt(λ[ℓ])*φ[:,ℓ]
    end

    return logk
end


function fwd(darcy::Darcy_params, v::AbstractVector{flt}) where {flt<:AbstractFloat}
    logk = get_logk(darcy,v)
    κ = exp.(logk)
    sol = Darcy_1D_solver(darcy,κ)
    y_locs = darcy.y_locs
    y = sol[y_locs]
    return y
end

function fwd_RLS(darcy::Darcy_params, v::AbstractVector{flt}) where {flt<:AbstractFloat}
    Hv = fwd(darcy, v)
    return [Hv; v]
end