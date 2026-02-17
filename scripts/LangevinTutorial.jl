using Random
using Statistics
using CairoMakie

## Setup
α = 0.4
Φ(x) = (x^2 - 2)^2 + α*x
∇Φ(x) = 4*x*(x^2 - 2) + α

function Langevin_EM(; Δt = 1e-3, J = 2_000_000, x0 = 0.0)
    x = x0
    σ = sqrt(2*Δt)
    out = Float64[]

    for j = 1:J
        x += -Δt*∇Φ(x) + σ*randn()
        push!(out,x)

    end
    return out
    
end

## Run 
Δt = 1e-3
J = 2_000_000
x0 = 2.0
xs = Langevin_EM(; Δt, J, x0)

## plots

# -----------------------------
# Build target density (normalized on a window)
# -----------------------------
trapz(x::AbstractVector, y::AbstractVector) = sum((y[1:end-1] .+ y[2:end]) .* diff(x)) / 2
xmin, xmax = -2.5, 2.5
grid = collect(range(xmin, xmax; length=2000))
p_unnorm = exp.(-Φ.(grid))
Z = trapz(grid, p_unnorm)
p = p_unnorm ./ Z

fig = Figure(size = (1000, 800))
# (1) Potential

ax1 = Axis(fig[1, 1], title = "Asymmetric double-well potential ϕ(x) = (x²-1)² + a x   (a=$α)",
           xlabel = L"x", ylabel = L"Φ(x)")
lines!(ax1, grid, Φ.(grid))


fig = Figure(size = (1000, 500))

# (3) Histogram + target density overlay
ax1 = Axis(fig[1, 1], title = "Stationary samples vs target density",
           xlabel = "x", ylabel = "density")

# Histogram normalized to integrate ~ 1
hist!(ax1, xs[1:2000000]; bins=120, normalization=:pdf)

# Overlay target density
lines!(ax1, grid, p; linewidth=5)

# A vertical line at x=0 to highlight asymmetry of mass
#vlines!(ax3, [0.0]; linestyle=:dash)


fig