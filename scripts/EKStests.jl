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

## EKS Implementation

@inline _colmean(V::AbstractMatrix) = vec(mean(V, dims=2)) # computes sample mean (column-wise)

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = _colmean(V)
    X = V .- μ
    return (X * X') / (J)
end

function _samplecrosscov(V1::AbstractMatrix, V2::AbstractMatrix)
    # computes sample cross-covariance between two ensembles with common ensemble size J
    J = size(V1, 2)
    μ₁ = _colmean(V1)
    μ₂ = _colmean(V2)
    V1c = V1 .- μ₁
    V2c = V2 .- μ₂
    return (V1c * V2c') / (J)
end

mutable struct EKSObj{T<:AbstractFloat, TI<:Int}
    V::Vector{Matrix{T}}
    y::Vector{T}
    Γ::Matrix{T}
    Ohist::Vector{Matrix{T}}
    μ₀::Vector{T}
    Γ₀::Matrix{T}
    J::TI; d::TI; n::TI
    iters::TI
    Δts::Vector{T}
end

function EKSObj(
    J::TI,
    μ₀::AbstractVector{T},
    Γ₀::AbstractMatrix{T},
    y::AbstractVector{T},
    Γ::AbstractMatrix{T},
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat, TI<:Int}
    d = length(μ₀)
    n = length(y)

    # sample initial ensemble
    prior = MvNormal(vec(μ₀), Symmetric(Matrix{T}(Γ₀)))
    V0 = rand(rng, prior, J)

    EKSObj{T, TI}(
        [Matrix{T}(V0)],
        Vector{T}(y),
        Matrix{T}(Γ),
        Matrix{T}[],
        Vector{T}(μ₀),
        Matrix{T}(Γ₀),
        J, d, n, 0, T[]
    )
end

function EKS_run!(
    obj::EKSObj{T},
    params,
    H::Function,
    iters::TI
) where {T<:AbstractFloat, TI<:Int}
    Hens = V -> H_ens(params, V, H)
    for _ in 1:iters
        EKS_step!(obj, Hens)
    end
    return obj
end



function EKS_step!(
    obj::EKSObj{T},
    Hens::Function
) where {T<:AbstractFloat}
    J, d, n = obj.J, obj.d, obj.n
    μ₀ = obj.μ₀
    Γ₀ = obj.Γ₀
    V = obj.V[end]
    O = Hens(V)

    V̄ = _colmean(V)
    Ō = _colmean(O)
    Cvv = _samplecov(V)
    Chh = _samplecov(O)

    E = O .- Ō # deviations from the mean
    R = O .- obj.y # residuals with data

    D = (1/J)*(E'*(obj.Γ\R)) # Gamma-weighted inner product

    Δt = 1/(norm(D) + 1e-8) # time step
    #Δt = 20/(obj.iters+1)
    #Δt = 0.3

    noise = MvNormal(Matrix(Hermitian(Cvv)))

    # Drift in state space
    A = Matrix{T}(I, d, d) + Δt * Cvv/Γ₀

    V_cent = V .- V̄

    rhs = V
          .- Δt * (V_cent * D)
          .+ Δt * (Cvv * (Γ₀ \ μ₀))
          + Δt * ((d + 1) / J) * V_cent # this is a correction not in the original EKS paper

    V_implicit = A\rhs

    V_new = V_implicit .+ sqrt(2*Δt).*rand(noise, J)

    push!(obj.V, V_new)
    push!(obj.Ohist, O)
    push!(obj.Δts, Δt)
    obj.iters += 1

    return nothing

end

## Let's test a linear problem
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
H = heat.H
H_s(::Nothing, v::AbstractVector) = H * v
J = 1000
V0 = rand(d, J)
#ekrmleobj = EKRMLEObj(V0, y_RLS, Γ_RLS)
μ_pr = vec(zeros(1, d))

## EKS straight from the source
dist = Parameterized(MvNormal(μ_pr, Symmetric(Γ_pr)))
constraint = repeat([no_constraint()], d)

prior = ParameterDistribution(dist, constraint, "v")
rng    = MersenneTwister(42)
J      = 2000
init   = EKP.construct_initial_ensemble(rng, prior, J)
eksobj = EKP.EnsembleKalmanProcess(init, y, Γ, Sampler(prior); rng=rng)
N_iter = 50
for n in 1:N_iter
    θ_ens = EKP.get_ϕ_final(prior, eksobj)            # d × J
    G_ens = [H_s(nothing, θ_ens[:, j]) for j in 1:J]
    g_ens = hcat(G_ens...)                            # n × J
    EKP.update_ensemble!(eksobj, g_ens)
end






##
eksobj = EKSObj(J,μ_pr,Γ_pr,y,Γ)
iters = 100
#EKRMLE_run!(ekrmleobj, nothing, H_RLS_s, iters)
EKS_run!(eksobj, nothing, H_s, iters)

## True posterior
Fish = (H'/Γ)*H
Γ_pos = (Fish + Γ_pr\I)\I
μ_pos = (Γ_pos*H'/Γ)*y

## 
C = _samplecov(EKP.get_ϕ_final(prior, eksobj))
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
#μ = _colmean(eksobj.V[end])
μ = EKP.get_ϕ_mean_final(prior, eksobj)
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title = L"\text{Posterior mean comparison}", titlesize=35)
lines!(ax1, μ_pos; linewidth=7, label=L"𝛍_\text{pos}", color=colors[2])
lines!(ax1, μ;linewidth=6, linestyle=:dash, label=L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]", color=colors[4])
axislegend(ax1; position=:rb, framevisible = false, labelsize=35)
display(fig)
#save("plots/mean_compare_HE.pdf", fig)

## Compare with black box sampler
bb_ens = rand(MvNormal(μ_pos, Symmetric(Γ_pos)), J)
m1 = 12
m2 = 150
bb_marg = bb_ens[[m1,m2],:]
PV_marg = (eksobj.V[end])[[m1,m2],:]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="RMLE in P-space")
scatter!(ax, bb_marg[1,:], bb_marg[2,:]; markersize=15, label="black-box", color=(colors[2], 0.30))
scatter!(ax, PV_marg[1,:], PV_marg[2,:]; markersize=15 ,label = "EKS",  color=(colors[4], 0.30))

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)



## EKS paper, 1st experiment
function p(x, v::AbstractVector)
    v1, v2 = v[1], v[2]
    return v2*x + exp(-v1)*(-((x^2)/2) + x/2)    
end

function 𝒢(x1, x2, v::AbstractVector)
    return vcat(p(x1, v),p(x2, v))
end

function 𝒢reg(x1, x2, v::AbstractVector)
    return vcat(𝒢(x1, x2, v), v)
    
end

# Setup
const x1, x2 = 1/4, 3/4
y = [27.5, 79.7]
d = 2; n = 2;
Γ = 0.1^2*Matrix{Float64}(I(2))
σ = 10
Γ_pr = σ^2*Matrix{Float64}(I(2))
Γ_RLS = Matrix{Float64}(I(n+d))
Γ_RLS[1:n, 1:n] .= Matrix{Float64}(Γ)
Γ_RLS[n+1:end, n+1:end] .= Matrix{Float64}(Γ_pr)
Γ_RLS = Symmetric(Γ_RLS)
y_RLS = vcat(Vector{Float64}(y), zeros(Float64, d))
J = 10000
V0 = rand(2, J)
ekrmleobj = EKRMLEObj(V0, y_RLS, Γ_RLS)
steps = 100
F_s(::Nothing, v::AbstractVector) =  𝒢reg(x1, x2, v)
EKRMLE_run!(ekrmleobj, nothing, F_s, steps)

## EKI
μ_pr = vec(zeros(1, d))
V0 = rand(MvNormal(μ_pr,Γ_pr),J)
ekiobj = EKIObj(V0, y_RLS, Γ_RLS, Γ_RLS)
EKI_run!(ekiobj, nothing, F_s, steps; flavor="vanilla")

## EKS from the source
FF_s(::Nothing, v::AbstractVector) = 𝒢(x1, x2, v)
prior = constrained_gaussian("theta", 0.0, σ, -Inf, Inf; repeats=d)
rng    = MersenneTwister(42)
J      = 1000
init   = EKP.construct_initial_ensemble(rng, prior, J)
eksobj = EKP.EnsembleKalmanProcess(init, y, Γ, Sampler(prior); rng=rng)
N_iter = 100
for n in 1:N_iter
    θ_ens = EKP.get_ϕ_final(prior, eksobj)            # d × J
    G_ens = [FF_s(nothing, θ_ens[:, j]) for j in 1:J]
    g_ens = hcat(G_ens...)                            # n × J
    EKP.update_ensemble!(eksobj, g_ens)
end



## EKS

eksobj = EKSObj(J,μ_pr,Γ_pr,y,Γ)
steps = 50
FF_s(::Nothing, v::AbstractVector) = 𝒢(x1, x2, v)
EKS_run!(eksobj, nothing, FF_s, steps)

## Plot
V = ekrmleobj.V[end]
V_EKI = ekiobj.V[end]
V_eks = EKP.get_ϕ_final(prior, eksobj)
fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="")
scatter!(ax, V_eks[1,:], V_eks[2,:]; markersize=15, label="EKS", color=(colors[3], 0.30))
scatter!(ax, V[1,:], V[2,:]; markersize=15 ,label = "EKRMLE",  color=(colors[4], 0.20))
scatter!(ax, V_EKI[1,:], V_EKI[2,:]; markersize=15 ,label = "EKI",  color=(colors[2], 0.30))

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)








## Setup 1D darcy
N, L = 256, 1.0
Nₖ = 32
d = Nₖ
Δ_obs = 5
σ = 0.1
α = 2.0
τ = 3.0
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



## EKS from the source
Γ = Matrix{Float64}(σ.^2*I(n))
prior = constrained_gaussian("theta", 0.0, 1.0, -Inf, Inf; repeats=d)
rng    = MersenneTwister(42)
J      = 1000
init   = EKP.construct_initial_ensemble(rng, prior, J)
eksobj = EKP.EnsembleKalmanProcess(init, y, Γ, Sampler(prior); rng=rng)
N_iter = 100
for n in 1:N_iter
    θ_ens = EKP.get_ϕ_final(prior, eksobj)            # d × J
    G_ens = [fwd(darcy, θ_ens[:, j]) for j in 1:J]
    g_ens = hcat(G_ens...)                            # n × J
    EKP.update_ensemble!(eksobj, g_ens)
end

## EKS
Γ = Matrix{Float64}(σ.^2*I(n))
Γ_pr = Matrix{Float64}(500*I(d))
μ_pr = vec(zeros(1, d))
J = 1000
eksobj = EKSObj(J,μ_pr,Γ_pr,y,Γ)
fwd_single(darcy, v::AbstractVector) = fwd(darcy, v)
steps = 50
EKS_run!(eksobj, darcy, fwd_single, steps)

## Plot EKS solution
μ = EKP.get_ϕ_mean_final(prior, eksobj)
#μ = vec(mean(eksobj.V[end],dims=2))
#μ = vec(eksobj.V[end][:,30])
logk_ens = get_logk(darcy, μ)

fig = Figure(size=(900,450))
ax = Axis(fig[1,1], xlabel="x", ylabel="", title="log permiability")
lines!(ax, X, logK; linewidth=10, linestyle=:solid, label=L"\log(a(𝐰;𝐯_\text{truth}))", color=colors[2])
lines!(ax, X, logk_ens[:]; linewidth=9, linestyle=:dash, label=L"\log(a(𝐰;\text{E}[𝐯^{(1:J)}_{\text{end}}] ))", color=colors[4])
axislegend(ax; position=:rb, labelsize=35, framevisible=false)
display(fig)

