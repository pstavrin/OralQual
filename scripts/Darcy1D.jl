## Darcy's law in 1D
using OralQual
using Distributions
using Random
using CairoMakie
using LinearAlgebra



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

# Setup
N, L = 256, 1.0
Nₖ = 15
d = Nₖ
Δ_obs = 5
σ = 0.1
α = 1.0
τ = 10.0
darcy = Darcy_params(N,L,Nₖ,Δ_obs,d,α,τ)
𝛉 = darcy.θ_true
logK = darcy.logk
y_locs = darcy.y_locs
n = darcy.n
X = darcy.X
κ = exp.(logK)
sol = Darcy_1D_solver(darcy,κ)

y_nonoise = sol[y_locs]
y = copy(y_nonoise)
for i = 1 : n
    y[i] += rand(Normal(0,σ^2))
end


fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=4, linestyle=:solid)

display(fig)
#save(joinpath("plots", "log_field.svg"), fig)



fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="Data")
lines!(ax, X, sol; linewidth=4, linestyle=:solid, label="True")
scatter!(ax, X[y_locs], y, color=:black, markersize=15, label="y")

display(fig)
#save(joinpath("plots", "Darcy_data.svg"), fig)


## EKRMLE
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

# Need to solve a Bayesian IP
# Treat prior as 𝒩(0, I)
Γ = σ.^2*I(n)
Γ_pr = I(d)

Γ_RLS = Matrix{Float64}(I(n+d))
Γ_RLS[1:n, 1:n] .= Matrix{Float64}(Γ)
Γ_RLS[n+1:end, n+1:end] .= Matrix{Float64}(Γ_pr)
Γ_RLS = Symmetric(Γ_RLS)
y_RLS = vcat(Vector{Float64}(y), zeros(Float64, d))
J = 5000
v₀ = rand(d, J)
ekrmleobj = EKRMLEObj(v₀, y_RLS, Γ_RLS)
fwd_RLS_single(darcy, v::AbstractVector) = fwd_RLS(darcy, v)
#v₀ = generate_Gaussian_noise(J, Γ_pr)
steps = 100
EKRMLE_run!(ekrmleobj, darcy, H_RLS, steps)


##
μ = vec(mean(ekrmleobj.V[end],dims=2))
logk_ens = get_logk(darcy, μ)

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=6, linestyle=:solid, label="Truth")
lines!(ax, X, logk_ens[:]; linewidth=6, linestyle=:dash, label="EKRMLE")
axislegend(ax; position=:lb, labelsize=20, framevisible=false)
display(fig)
#save(joinpath("plots", "Darcy_EKRMLE.svg"), fig)





## Need to solve this via MCMC as well to compare

function log_likelihood(y::Array{flt,1}, Hv::Array{flt,1}, Γ::Array{flt,2}) where {flt<:Real}
    return -0.5*(y-Hv)'*(Γ\(y-Hv))
    
end

function RWMH(log_likelihood::Function, v₀, Δt::flt, iters::int) where {flt<:Real,int<:Int}
    d = length(v₀)
    V = zeros(d,iters+1)
    loglikes = zeros(iters+1)
    
    V[:,1] = v₀
    Hv = fwd_RLS(v₀,φ,λ,N,Δx,f,y_locs)
    loglikes[1] = log_likelihood(Hv)

    for i = 2 : iters+1
        v_prev = V[:,i-1]
        v = v_prev + Δt * rand(Normal(0,1), d)
        Hv = fwd_RLS(v,φ,λ,N,Δx,f,y_locs)
        loglikes[i] = log_likelihood(Hv)
        α = min(1.0, exp(loglikes[i] - loglikes[i-1]))
        u = rand(Uniform(0,1))

        if α >= u
            # accept choice
            V[:,i] = v
        else
            # reject choice
            V[:,i] = V[:,i-1]
            loglikes[i] = loglikes[i-1]
        end
    end


    return V

end


log_lik(Hv) = log_likelihood(y_RLS[:], Hv, Γ_RLS)
vMC₀ = rand(Normal(0,1), Nₖ)
Δt = 1e-3
MCMC_iters = 2000000
V_MCMC = RWMH(log_lik, vMC₀, Δt, MCMC_iters)


##
burn_in = Int(1.5e6)
μ_MCMC = mean(V_MCMC[:,burn_in:end],dims=2)
logk_MCMC = zeros(N,1)

for ℓ = 1 : d
    logk_MCMC .+= μ_MCMC[ℓ]*sqrt(λ[ℓ])*φ[:,ℓ]
end



fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=6, linestyle=:solid, label="True")
lines!(ax, X, logk_ens[:]; linewidth=6, linestyle=:dash, label="EKRMLE")
lines!(ax, X, logk_MCMC[:]; linewidth=6, linestyle=:dot, label="RWMH")
axislegend(ax; position=:lb, labelsize=20, framevisible=false)
display(fig)
save(joinpath("plots", "Darcy_EKRMLE_RWMH.svg"), fig)

## Plot some marginals
m1 = 19
m2 = 3
burn_in = Int(1.8e6)
burn_out = Int(2e6)
V_marg = V[[m1,m2],:,end]
MCMC_marg = V_MCMC[[m1,m2],burn_in:burn_out]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=15, label="EKRMLE")
scatter!(ax, MCMC_marg[1,:], MCMC_marg[2,:]; markersize=15, marker=:cross ,label = "RWMH")

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)
save(joinpath("plots", "Darcy_marginals.png"), fig)

## Plot uncertainty?
    C = sqrt.(diag(Γ_store[:,:,end]))
C_MCMC = sqrt.(diag(cov(V_MCMC[:,burn_in:burn_out]')))



XX = 1:1:d
fig = Figure(size=(1800,600))
ax = Axis(fig[1,1], xlabel="index", ylabel="", title="uncertainty")
scatterlines!(ax, 𝛉, linewidth=4, markersize=15, label="Truth", color=:black)
scatterlines!(ax, μ[:], linewidth=3, markersize=15, marker=:cross, linestyle=:dash, label="EKRMLE")
scatterlines!(ax, μ_MCMC[:], linewidth=3, markersize=15, marker=:cross, linestyle=:dash, label="RWMH")
band!(ax,XX, μ[:] .- 3 .* C[:], μ[:] .+ 3 .* C[:]; color=(:purple,0.18))
band!(ax,XX, μ_MCMC[:] .- 3 .* C_MCMC[:], μ_MCMC[:] .+ 3 .* C_MCMC[:]; color=(:red,0.18))
axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)