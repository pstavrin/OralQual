## Sampling non-Gaussian distributions
using OralQual
using Distributions
using Random
using CairoMakie
using LinearAlgebra
using StatsBase
using KernelDensity

## Setup


function ℋ(v₁,v₂) # Banana Distribution
    return (1)*(v₂) - (1/10)*(v₁)^2
end



function ℋ(v₁,v₂,v₃)
    β = 2
    γ = 1.0
    return [v₁, v₂ - β*sin(v₁^2), v₃ + γ*v₁ + v₂^2]
end




function ℋ(v₁,v₂) # Donut
    return (v₁^2+v₂^2-25)^2
end



function ℋ(v₁,v₂) # Rosenbrock
    β = sqrt(100.0)
    α = 1.0
    return [sqrt(2)*(v₁-α), sqrt(2*β)*(v₂-v₁^2)]
end


function ℋ(v₁,v₂) # Banana Distribution
    β = -0.6
    return [v₁, v₂ - β*(v₁^2-1)]
end



function ℋ(v₁,v₂) # split Distribution
    β = 0.5
    return [v₁^2, v₂ - β*(v₁^2 - 1)]
end



function ℋ(v₁,v₂) # sin Distribution
    β = 5
    return [v₁^2, v₂ - β*sin(π*v₁)]
end




function ℋ(v₁,v₂) # funnel Distribution
    s = 2
    return [v₁, exp(-(s/2)*v₁)*v₂]
end




function ℋ(v₁,v₂) # Banana Distribution
    β = 0.9
    return [v₁, v₂ - β*(v₁^2-1)]
end


function init_rect(J; x1=-1,x2=1,y1=-2,y2=2)
    x = x1 .+ (x2-x1) .*rand(J)
    y = y1 .+ (y2-y1) .*rand(J)
    return vcat(x',y')
    
end

y = [0.0,0.0]
Γ = (1/1)*I(2)
J = 500_000
steps = 50
#V₀ = init_rect(J; x1=-10,x2=10,y1=-3,y2=5)
V₀ = randn(2,J)
ekrmleobj = EKRMLEObj(V₀, y, Γ)
H_single(::Nothing,v::AbstractVector) = ℋ(v[1],v[2])
EKRMLE_run!(ekrmleobj, nothing, H_single, steps)

## Plots
V = ekrmleobj.V[end]
fig = Figure(size = (600, 600))
ax = Axis(fig[1, 1]; title="Ensemble start", xlabel="v₁", ylabel="v₂")

# ---- Contours of ℋ over data extents (+ padding) ----
padx = 5.2; pady = 5.2
ys = range(minimum(V[1,:]) - padx, maximum(V[1,:]) + padx; length=1000)  # x from V[1,...]
xs = range(minimum(V[2,:]) - pady, maximum(V[2,:]) + pady; length=1000)  # y from V[2,...]
xs = range(-6,6, 1000)
ys = range(-22,7,1000)

# Makie expects Z as (length(ys), length(xs)) and ℋ(x,y)
#Z = [exp.(-ℋ(y, x).^2) for y in ys, x in xs]
ϕ(u) = exp(-(1/2)*(u^2))/(sqrt(2*pi))
ρ(x, y) = ϕ(x)*ϕ(y - (0.9)*(x^2-1.0))
#ρ(x, y) = (2*sqrt(100))*ϕ(sqrt(2)*(-x+1))*ϕ(sqrt(2*100)*(y-x^2))
s = 2
#ρ(x, y) = (1/(2*pi))*exp(-(x^2)/2)*exp(-s*x/2)*exp(-0.5*(y^2)*exp(-s*x))
Z = [ρ(x, y) for x in xs, y in ys]

# Try auto levels first to ensure something shows up
#contour!(ax, xs, ys, Z; levels=10, linewidth=1.5, color=:gray)

Zlog = log10.(Z .+ 1e-300)   # avoid -Inf
heatmap!(ax, xs, ys, Zlog;
    colormap = :magma,       # or :viridis, :plasma, etc.
    colorrange = (-8, 0),
    interpolate = true
)

# If you want explicit levels later:
levels = [1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 50 ,100, 1000]
contour!(ax, xs, ys, Z; levels=levels, linewidth=2.5, color="#51127C")


#scatter!(ax, V[1,:], V[2,:], markersize = 10, color = ("#FEBB81",0.4))
#scatter!(ax, V[1,:], V[2,:], markersize = 10, color = ("#5DCEAF",0.4))
#scatter!(ax, mean(V[1,:]), mean(V[2,:]), markersize=20)



display(fig)
##
V = ekrmleobj.V[end]
xv = V[1, :]
yv = V[2, :]

k = kde((xv, yv))

xsk = k.x
ysk = k.y
Zens  = k.density          # size should be (length(xs), length(ys))

# If Makie complains about dimensions, try: Z = permutedims(Z)  (see note below)

# Choose contour levels (log-spaced looks like your example)
zmax = maximum(Zens)
zmin = zmax * 1e-4       # controls how far out you show tails
levels = exp10.(range(log10(zmin), log10(zmax); length=10))

# --- Plot ---
fig = Figure(size = (900, 900))
ax  = Axis(fig[1,1])

contourf!(ax, xsk, ysk, Zens;
    levels = levels,
    colormap = :magma,
    extendhigh = :auto,
    extendlow  = :auto
)
hidedecorations!(ax)
hidespines!(ax)


fig

##


Z = [ρ(x, y) for x in xsk, y in ysk]
zmax = maximum(Z)
zmin = zmax * 1e-4 
levels = exp10.(range(log10(zmin), log10(zmax); length=10))

# --- Plot ---
fig = Figure(size = (900, 900))
ax  = Axis(fig[1,1])

contourf!(ax, xsk, ysk, Z;
    levels = levels,
    colormap = :magma,
    extendhigh = :auto,
    extendlow  = :auto
)
hidedecorations!(ax)
hidespines!(ax)

#scatter!(ax, V[1,:], V[2,:], markersize = 10, color = ("#FEBB81",0.4))

fig

