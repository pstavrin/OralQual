## Sampling non-Gaussian distributions
using OralQual
using Distributions
using Random
using CairoMakie
using LinearAlgebra

## Setup


function ℋ(v₁,v₂) # Donut
    return (v₁^2+v₂^2-25)^2
end


function ℋ(v₁,v₂) # Banana Distribution
    return (1)*(v₂) - (1/2)*(v₁)^2
end


function init_rect(J; x1=-1,x2=1,y1=-2,y2=2)
    x = x1 .+ (x2-x1) .*rand(J)
    y = y1 .+ (y2-y1) .*rand(J)
    return vcat(x',y')
    
end

y = [0.0]
Γ = (1/2)*I(1)
J = 5000
steps = 500
V₀ = init_rect(J; x1=-7,x2=10,y1=-10,y2=15)
ekrmleobj = EKRMLEObj(V₀, y, Γ)
H_single(::Nothing,v::AbstractVector) = [ℋ(v[1],v[2])]
EKRMLE_run!(ekrmleobj, nothing, H_single, steps)

## Plots
V = ekrmleobj.V[end]
fig = Figure(size = (600, 600))
ax = Axis(fig[1, 1]; title="Ensemble start", xlabel="v₁", ylabel="v₂")

# ---- Contours of ℋ over data extents (+ padding) ----
padx = 0.2; pady = 0
ys = range(minimum(V₀[1,:]) - padx, maximum(V₀[1,:]) + padx; length=400)  # x from V[1,...]
xs = range(minimum(V₀[2,:]) - pady, maximum(V₀[2,:]) + pady; length=400)  # y from V[2,...]

# Makie expects Z as (length(ys), length(xs)) and ℋ(x,y)
Z = [exp.(-ℋ(y, x).^2) for y in ys, x in xs]

# Try auto levels first to ensure something shows up
#contour!(ax, xs, ys, Z; levels=10, linewidth=1.5, color=:gray)

# If you want explicit levels later:
levels = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5, 10, 50 ,100]
contour!(ax, ys, xs, Z; levels=levels, linewidth=4, color=:black)

scatter!(ax, V[1,:], V[2,:], markersize = 10, color = (:blue,0.1))
#scatter!(ax, mean(V[1,:]), mean(V[2,:]), markersize=20)
display(fig)


