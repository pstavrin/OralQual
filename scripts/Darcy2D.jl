using OralQual
using LinearAlgebra
using Random
using CairoMakie
using Distributions
using SparseArrays
using ColorSchemes
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
include("PrettyPlots.jl")


## Setup
N, L = 80, 1.0
obs_ΔN = 8
α = 2.0
τ = 10.0
N_KL = 32
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
ax.title = L"\log(a(𝐱;\textbf{v}_{\text{truth}}))"
ax.titlesize = 40
display(fig)
save("plots/Darcy_2D_truth_d32.pdf",fig)
#save("plots/Darcy_2D_truth.svg",fig)

## Plot solution
fig, ax = plot_field(darcy, h, true)
ax.title = L"p(𝐱)"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_p_manypoints.pdf",fig)
#save("plots/Darcy_2D_p.svg",fig)


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
ax.title = L"\log(a(𝐱;𝐯_{\text{EKRMLE}}))"
ax.titlesize = 40
display(fig)
save("plots/Darcy_2D_ekrmle_d32.pdf",fig)
#save("plots/Darcy_2D_ekrmle.svg",fig)

## Compare with ground truth
plot_field_sbs(darcy, darcy.logk_2d, logk_EKRMLE)

## Plot error
fig, ax = plot_field(darcy, abs.(darcy.logk_2d - logk_EKRMLE))
ax.title = L"\text{Absolute error}"
ax.titlesize = 40
display(fig)
save("plots/Darcy_2D_error_d32.pdf",fig)
#save("plots/Darcy_2D_error.svg",fig)


## Uncertainty plot
σ = sqrt.(diag(_samplecov(ekrmleobj.V[end])))   # or however you compute it
upper = μ .+ 2 .* σ
lower = μ .- 2 .* σ
colors = my_custom_dark_theme.palette.color[]

fig = themed_figure(; dark=false, size=(1200, 300)) do fig
    ax1 = Axis(fig[1, 1],
        title     = L"\text{Uncertainty quantification } (d=32)",
        titlesize = 40,
    )

    band!(ax1, 1:length(μ), lower, upper;
        color = (colors[3], 0.20), 
        label = L"\pm 2\sigma",
    )

    scatterlines!(ax1, darcy.θ_true;
        linewidth   = 8,
        markersize  = 15,
        label       = L"𝐯_\text{truth}",
        color       = colors[4],
    )

    scatterlines!(ax1, μ;
        linewidth   = 7,
        linestyle   = :dash,
        markersize  = 15,
        label       = L"𝐯_\text{EKRMLE}",
        color       = colors[3],
    )

    legend = Legend(fig, ax1; framevisible=false, labelsize=40)

    fig[1, 2] = legend    # <-- place legend in separate column

    colsize!(fig.layout, 2, Auto())   # auto-size legend column

    #axislegend(ax1; position=:rb, framevisible=false, labelsize=35)
    fig   # important: return the figure from the do-block
end

save("plots/Darcy_2D_d32_uncertain.pdf",fig)






## EKS
dist = Parameterized(MvNormal(vec(zeros(1, d)), Symmetric(Γ_pr)))
constraint = repeat([no_constraint()], d)
prior = ParameterDistribution(dist, constraint, "v")
init   = EKP.construct_initial_ensemble(rng, prior, J)
eksobj = EKP.EnsembleKalmanProcess(init, y, Γ, Sampler(prior))
for n in 1:steps
    θ_ens = EKP.get_ϕ_final(prior, eksobj)            # d × J
    G_ens = [fwd_2D(darcy, θ_ens[:, j]) for j in 1:J]
    g_ens = hcat(G_ens...)                            # n × J
    EKP.update_ensemble!(eksobj, g_ens)
end


## EKS log field
μ_EKS = vec(mean(EKP.get_ϕ_final(prior, eksobj),dims=2))
logk_EKS = get_logk_2D(darcy, μ_EKS)
fig, ax = plot_field(darcy, logk_EKS, false)
ax.title = L"\log(a(𝐰; 𝐯_{\text{EKS}}))"
ax.titlesize = 40
display(fig)

##
plot_field_sbs(darcy, logk_EKRMLE, logk_EKS; titles=("EKRMLE", "EKS"))

## EKS error
fig, ax = plot_field(darcy, abs.(darcy.logk_2d - logk_EKS))
ax.title = L"\text{Absolute error EKS}"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_error.pdf",fig)
#save("plots/Darcy_2D_error.svg",fig)



## Plot some marginals
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
m1 = 12
m2 = 5
V_marg = ekrmleobj.V[end][[m1,m2],:]
EKS_marg = EKP.get_ϕ_final(prior, eksobj)[[m1,m2],:]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=15, label="EKRMLE", color=(colors[2], 0.50))
scatter!(ax, EKS_marg[1,:], EKS_marg[2,:]; markersize=15 ,label = "EKS", color=(colors[4], 0.50))

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)