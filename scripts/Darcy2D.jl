using OralQual
using LinearAlgebra
using Random
using CairoMakie
using Distributions
using SparseArrays


## Setup
N, L = 80, 1.0
obs_ΔN = 8
α = 2.0
τ = 10.0
N_KL = 10
σ₀ = 1.0
d = N_KL
noise_level = 0.05 # 5% of output
darcy = Darcy_params_2D(N, L, N_KL, obs_ΔN, obs_ΔN, d, α, τ, σ₀)
κ = exp.(darcy.logk_2d)
h = Darcy_2D_solver(darcy, κ)
y_nonoise = get_data(darcy, h)
# create noisy observations
y = copy(y_nonoise)
for i = 1:darcy.n
    noise = rand(Normal(0, noise_level*y[i]))
    y[i] += noise
end
Γ = Array(Diagonal(fill(1.0, length(y))))

## Plot truth
fig, ax = plot_field(darcy, darcy.logk_2d, false)
ax.title = L"\log(a(𝐰;\textbf{v}_{\text{truth}}))"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_truth.pdf",fig)
save("plots/Darcy_2D_truth.svg",fig)

## Plot solution
fig, ax = plot_field(darcy, h, true)
ax.title = L"p(𝐰)"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_p.pdf",fig)
save("plots/Darcy_2D_p.svg",fig)


## EKRMLE
# Need to solve a Bayesian IP
# Treat prior as 𝒩(0, I)
n, d = length(y), darcy.d
Γ_pr = I(d)
Γ_RLS = Matrix{Float64}(I(n+d))
Γ_RLS[1:n, 1:n] .= Matrix{Float64}(Γ)
Γ_RLS[n+1:end, n+1:end] .= Matrix{Float64}(Γ_pr)
Γ_RLS = Symmetric(Γ_RLS)
y_RLS = vcat(Vector{Float64}(y), zeros(Float64, d))
J = 500
v₀ = rand(d, J)
ekrmleobj = EKRMLEObj(v₀, y_RLS, Γ_RLS)
fwd_RLS_single(darcy, v::AbstractVector) = fwd_RLS_2D(darcy, v)
steps = 25
EKRMLE_run!(ekrmleobj, darcy, fwd_RLS_single, steps)


## Plot EKRMLE field
μ = vec(mean(ekrmleobj.V[end],dims=2))
logk_EKRMLE = get_logk_2D(darcy, μ)
fig, ax = plot_field(darcy, logk_EKRMLE, false)
ax.title = L"\log(a(𝐰;\text{E}[𝐯^{(1:J)}_{\text{end}}] ))"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_ekrmle.pdf",fig)
save("plots/Darcy_2D_ekrmle.svg",fig)

## Compare with ground truth
plot_field_sbs(darcy, darcy.logk_2d, logk_EKRMLE)

## Plot error
fig, ax = plot_field(darcy, abs.(darcy.logk_2d - logk_EKRMLE))
ax.title = L"\text{Absolute error}"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_error.pdf",fig)
save("plots/Darcy_2D_error.svg",fig)
## Plot some marginals
m1 = 5
m2 = 10
burn_in = Int(1.8e6)
burn_out = Int(2e6)
V_marg = ekrmleobj.V[end][[m1,m2],:]
#MCMC_marg = V_MCMC[[m1,m2],burn_in:burn_out]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=15, label="EKRMLE")
#scatter!(ax, MCMC_marg[1,:], MCMC_marg[2,:]; markersize=15, marker=:cross ,label = "RWMH")

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)