using OralQual
using LinearAlgebra
using CairoMakie
using Distributions
using Random
using ColorSchemes
using MAT
using SparseArrays
using ControlSystems

@inline colmean(V::AbstractMatrix) = vec(mean(V, dims=2))

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

## Setup
path = "data/heat-cont.mat"
n = 100
Δt = 1e-3
T_stop = 10.0
heat = heat_eq_params(n, Δt, T_stop, path)
d = heat.d

## noisy data
Γ_pr = lyap(Matrix(heat.A), I(heat.d))
truth = rand(MvNormal(vec(zeros(1,d)),Symmetric(Γ_pr)))
sol_nonoise = solve_HE(heat, truth)
h, Δt = heat.h, heat.Δt
meas_idx = Int.(round.(heat.T_meas ./Δt))
y_nonoise = sol_nonoise[meas_idx]
σ = 0.08*maximum(abs.(y_nonoise))
γ = fill(σ^2, n)
Γ = γ .* I(n)
ε = rand(MvNormal(vec(zeros(1,n)),Γ))
y = y_nonoise + ε

## Plot data
fig = Figure()
ax = Axis(fig[1,1],
    xlabel=L"\text{Time}",
)

scatter!(ax, heat.TT, vec(sol_nonoise), label= L"\text{solution}")
scatter!(ax, heat.T_meas, y, marker=:cross, label = L"y")
axislegend(position =:lb, labelsize = 20)
display(fig)

## inversion
# We solve the Bayesian IP with prior 𝒩(0,Γ_pr)
Γ_RLS = Matrix{Float64}(I(n+d))
Γ_RLS[1:n, 1:n] .= Matrix{Float64}(Γ)
Γ_RLS[n+1:end, n+1:end] .= Matrix{Float64}(Γ_pr)
Γ_RLS = Symmetric(Γ_RLS)
y_RLS = vcat(Vector{Float64}(y), zeros(Float64, d))
H_RLS = vcat(heat.H, I(d))
H_RLS_s(::Nothing, v::AbstractVector) = H_RLS * v
J = 10000
V0 = rand(d, J)
ekrmleobj = EKRMLEObj(V0, y_RLS, Γ_RLS)
iters = 30
EKRMLE_run!(ekrmleobj, nothing, H_RLS_s, iters)

## True posterior
H = heat.H
Fish = (H'/Γ)*H
Γ_pos = (Fish + Γ_pr\I)\I
μ_pos = (Γ_pos*H'/Γ)*y

## 
C = _samplecov(ekrmleobj.V[end])
## Covariance comparison
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title=L"\text{Cov}[\textbf{v}_\text{end}^{(1:J)}]", titlesize=35)
ax1.yreversed=true
ax2 = Axis(fig[1, 3], title = L"\textbf{Γ}_\text{pos}", titlesize=35)
ax2.yreversed=true
hm1 = heatmap!(ax1, C; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, Γ_pos; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)
#save("plots/cov_compare_HE.pdf", fig)

## Mean comparison
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
μ = colmean(ekrmleobj.V[end])
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title = L"\text{Posterior mean comparison}", titlesize=35)
lines!(ax1, μ_pos; linewidth=7, label=L"𝛍_\text{pos}", color=colors[2])
lines!(ax1, μ;linewidth=6, linestyle=:dash, label=L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]", color=colors[4])
axislegend(ax1; position=:rb, framevisible = false, labelsize=35)
display(fig)
#save("plots/mean_compare_HE.pdf", fig)