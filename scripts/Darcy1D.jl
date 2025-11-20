## Darcy's law in 1D
using OralQual
using Distributions
using Random
using CairoMakie
using LinearAlgebra
using ColorSchemes

## Setup
N, L = 256, 1.0
Nₖ = 32
d = Nₖ
Δ_obs = 5
σ = 0.1
α = 2.0
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

## Plot log field
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=7, linestyle=:solid, color=colors[2])

display(fig)
#save(joinpath("plots", "log_field.svg"), fig)


## Plot solution and measurements
fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="Data")
lines!(ax, X, sol; linewidth=7, linestyle=:solid, label="True", color=colors[2])
scatter!(ax, X[y_locs], y, markersize=20, label="y", color=colors[4])

display(fig)
#save(joinpath("plots", "Darcy_data.svg"), fig)


## EKRMLE
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
EKRMLE_run!(ekrmleobj, darcy, fwd_RLS_single, steps)


##
μ = vec(mean(ekrmleobj.V[end],dims=2))
logk_ens = get_logk(darcy, μ)

fig = Figure(size=(900,450))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=10, linestyle=:solid, label=L"\log(a(𝐰;𝐯_\text{truth}))", color=colors[2])
lines!(ax, X, logk_ens[:]; linewidth=9, linestyle=:dash, label=L"\log(a(𝐰;\text{E}[𝐯^{(1:J)}_{\text{end}}] ))", color=colors[4])
axislegend(ax; position=:rb, labelsize=35, framevisible=false)
display(fig)
#save(joinpath("plots", "Darcy_EKRMLE.svg"), fig)





## Need to solve this via MCMC as well to compare

function log_likelihood(y::AbstractVector{flt}, Hv::AbstractVector{flt}, Γ::AbstractMatrix{flt}) where {flt<:AbstractFloat}
    return -0.5*(y-Hv)'*(Γ\(y-Hv))
    
end

function RWMH(darcy::Darcy_params,log_likelihood::Function, v₀::AbstractVector{T}, iters::TI) where {T<:AbstractFloat,TI<:Int}
    d = length(v₀)
    V = zeros(d,iters+1)
    loglikes = zeros(iters+1)
    
    V[:,1] = v₀
    Hv = fwd_RLS(darcy, v₀)
    loglikes[1] = log_likelihood(Hv)

    for i = 2 : iters+1
        v_prev = V[:,i-1]
        v = v_prev + Δt * rand(Normal(0,1), d)
        Hv = fwd_RLS(darcy, v)
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
vMC₀ = vec(rand(Normal(0,1), Nₖ))
Δt = 1e-3
MCMC_iters = 2000000
V_MCMC = RWMH(darcy, log_lik, vMC₀, MCMC_iters)


## plot vs MCMC
burn_in = Int(1.8e6)
μ_MCMC = vec(mean(V_MCMC[:,burn_in:end],dims=2))
logk_MCMC = get_logk(darcy, μ_MCMC)
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=8)]
fig = Figure(size=(900,400))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=10, linestyle=:solid, label=L"\log(a(𝐰;𝐯_\text{truth}))", color=colors[3])
lines!(ax, X, logk_ens[:]; linewidth=9, linestyle=:dash, label=L"\log(a(𝐰;𝐯_\text{EKRMLE}))", color=colors[5])
lines!(ax, X, logk_MCMC[:]; linewidth=9, linestyle=:dot, label=L"\log(a(𝐰;𝐯_\text{RWMH}))", color=colors[7])
axislegend(ax; position=:rb, labelsize=35, framevisible=false)
display(fig)
#save(joinpath("plots", "Darcy_EKRMLE_RWMH.svg"), fig)

## Plot some marginals
m1 = 1
m2 = 7
burn_in = Int(1.8e6)
burn_out = Int(2e6)
V_marg = ekrmleobj.V[end][[m1,m2],:]
MCMC_marg = V_MCMC[[m1,m2],burn_in:burn_out]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=20, label="EKRMLE", color=(colors[3], 0.20))
scatter!(ax, MCMC_marg[1,:], MCMC_marg[2,:]; markersize=10,label = "RWMH", color=(colors[6], 0.05))

axislegend(ax; position=:rb, framevisible = false, labelsize=35)

display(fig)
#save(joinpath("plots", "Darcy_marginals.png"), fig)

## Plot uncertainty?
@inline _colmean(V::AbstractMatrix) = vec(mean(V, dims=2))
function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = _colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end



C = sqrt.(diag(_samplecov(ekrmleobj.V[end])))
C_MCMC = sqrt.(diag(_samplecov(V_MCMC)))



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