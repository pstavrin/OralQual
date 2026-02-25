using OralQual
using Random
using LinearAlgebra
using DifferentialEquations
using Distributions
using CairoMakie

function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ*(y - x)
    du[2] = x*(ρ - z) - y
    du[3] = x*y - β*z
    return nothing
end

# Observation operator
# Full-state:
H(u) = u
# Only x:
# H(u) = [u[1]]
# x and z:
# H(u) = [u[1], u[3]]

## Data generation
v_true = [10.0, 28.0, 8/3]
v0_true = [1.0, 1.0, 1.0]

T  = 2.0
dt = 0.01
t_obs = 0.0:0.05:T         # observation times
γ = 2.0                    # noise std
Γ = γ^2 * I(length(H(v0_true)))  # obs covariance at each time

prob_true = ODEProblem(lorenz63!, v0_true, (0.0, T), v_true)
sol_true  = solve(prob_true, Tsit5(); saveat=t_obs)

# Stack clean observations
y_clean = vcat([H(sol_true(t)) for t in t_obs]...)  # length = m*K

# Add i.i.d. Gaussian noise per observation time
# (Γ applies per time; block-diagonal overall)
y = copy(y_clean)
n = length(y)
for k in 1:n
    y[k] += γ * randn()
end

## Forward model for EKRMLE
function H_Lor(nothing, v; v0=v0_true, t_obs=t_obs)
    prob = ODEProblem(lorenz63!, v0, (t_obs[1], t_obs[end]), v)
    sol  = solve(prob, Tsit5(); saveat=t_obs, abstol=1e-9, reltol=1e-9)
    return vcat([H(sol(t)) for t in t_obs]...)
end

H_Lor(nothing, v_true)

## EKRMLE
Γ = (1/1)*I(n)
J = 100
V₀ = [15.0, 32.0, 1/3] .+ γ .*randn(3, J)
obj = EKRMLEObj(V₀, y, Γ)
## Run
steps = 10
EKRMLE_run!(obj, nothing, H_Lor, steps)

##
V = obj.V[end]
μ = vec(mean(V, dims = 2))
C = (V .- μ) * (V .- μ)' /(J-1)
d = length(μ)
## Plots


# mean and std from ensemble covariance at each iteration
μ = zeros(d, steps)
σ = zeros(d, steps)

for n in 1:steps
    V = obj.V[n]
    μ[:, n] = vec(mean(V; dims=2))

    # sample covariance across particles
    C = (V .- μ[:,n]) * (V .- μ[:,n])' /(J-1)
    σ[:, n] .= sqrt.(diag(C))
end

names = ["σ", "ρ", "β"]  # adjust if you inferred more params
kband = 3.0              # 2σ band; use 1.0 for 1σ

fig = Figure(size=(900, 700))
for i in 1:d
    ax = Axis(fig[i, 1], xlabel="EKRMLE iteration", ylabel=names[i])

    lo = μ[i, :] .- kband .* σ[i, :]
    hi = μ[i, :] .+ kband .* σ[i, :]

    band!(ax, 1:steps, lo, hi, color="#FEBB81")
    lines!(ax, 1:steps, μ[i, :], linewidth=7, color="#51127C")
    hlines!(ax, v_true[i], linestyle=:dash, color=:black, linewidth=4)
end

fig