## Sampling non-Gaussian distributions
using OralQual
using Distributions
using Random
using CairoMakie
using LinearAlgebra
using StatsBase
using KernelDensity
using Flux
## Setup


function ℋ(v₁,v₂) # Banana Distribution
    return (1)*(v₂) - (1/10)*(v₁)^2
end



function ℋ(v₁,v₂,v₃)
    β = 2
    γ = 1.0
    return [v₁, v₂ - β*sin(v₁^2), v₃ + γ*v₁ + v₂^2]
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









function ℋ(v₁,v₂) # split Distribution
    δ = 20
    κ = 1/20
    return [v₁ - δ*tanh(κ*v₁), v₂]
end



function ℋ(v₁,v₂) # Donut
    return (v₁^2+v₂^2-49)^2
end


function ℋ(v₁,v₂,v₃) # sin/banana mix Distribution
    A = 1/2
    ω = 2*π
    β = 0.9
    return [v₁, v₂ - A*sin.(ω*v₁), v₃ - β*(v₁.^2-1)]
end



function ℋ(v₁,v₂) # tanh Distribution
    return [v₁+3*tanh(v₁), v₂ + 2*tanh(v₂)+ 10*tanh(v₁)]
end





function ℋ(v₁,v₂) # funnel Distribution
    s = 2
    return [v₁, exp(-(s/2)*v₁)*v₂]
end




function ℋ(v₁,v₂) # sin Distribution
    A = 3
    ω = 2*π
    return [v₁, v₂ - A*sin(ω*v₁)]
end



function ℋ(v₁,v₂) # Banana Distribution
    α = 1
    β = 0.9
    γ = 1
    return [α*v₁, γ*v₂ - β*(v₁^2-1)]
end





function ℋ(v₁,v₂) # iverse map of Banana Distribution
    α = 2
    β = -0.9
    γ = 1/2
    return [v₁/α, (1/γ)*(v₂ + β*((v₁/α)^2 - 1))]
end




function ℋ(v₁,v₂) # inverse map for sin Distribution
    A = 1
    ω = 2*π
    return [v₁, v₂ + A*sin(ω*v₁)]
end





function ℋ(v₁,v₂) # inverse map of funnel Distribution
    s = 2
    return [v₁, v₂*exp(s*v₁)]
end


function init_rect(J; x1=-1,x2=1,y1=-2,y2=2)
    x = x1 .+ (x2-x1) .*rand(J)
    y = y1 .+ (y2-y1) .*rand(J)
    return vcat(x',y')
    
end

y = [0.0,0.0]
Γ = (1/1)*I(2)
J = 10_000
steps = 2000
#V₀ = init_rect(J; x1=-8,x2=10,y1=-5,y2=10)
V₀ = 0.1 .*randn(2,J)
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
α = 2; β = -0.9; γ = 1/2
ρ(x, y) = (1/(α*γ))*ϕ(x/α)*ϕ((1/γ)*(y+β*((x/α)^2-1)))
#ρ(x, y) = (2*sqrt(100))*ϕ(sqrt(2)*(-x+1))*ϕ(sqrt(2*100)*(y-x^2))
s = 2
#ρ(x, y) = (1/(2*pi))*exp(-(x^2)/2)*exp(-s*x/2)*exp(-0.5*(y^2)*exp(-s*x))
ρ(x, y) = exp(s*x)*ϕ(x)*ϕ(y*exp(s*x))
A = 1; ω = (2)*π;
#ρ(x, y) = (1/(2π))*exp(-0.5*x^2 - 0.5*(y-A*sin(ω*x))^2)
#ρ(x, y) = ϕ(x)*ϕ(y + A*sin(ω*x))
#ρ(x, y) = exp(-(x^2+y^2-25)^2)
Z = [ρ(x, y) for x in xs, y in ys]

# Try auto levels first to ensure something shows up
#contour!(ax, xs, ys, Z; levels=10, linewidth=1.5, color=:gray)

Zlog = log10.(Z .+ 1e-300)
heatmap!(ax, xs, ys, Zlog;
    colormap = :magma,
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

contourf!(ax, xsk,  ysk, Z;
    levels = levels,
    colormap = :magma,
    extendhigh = :auto,
    extendlow  = :auto
)
hidedecorations!(ax)
hidespines!(ax)

scatter!(ax, V[1,:], V[2,:], markersize = 15, color = ("#FEBB81",0.4))

display(fig)
#save("plots/FunnelDensity1WithEnsemble.svg",fig)

## try plotting conditionals
ε = ekrmleobj.Yrand

function conditional_groups(ε; nbins=50)
    e = vec(ε[1, :])                # condition on ε₁
    bins = range(minimum(e), maximum(e), length=nbins+1)

    groups = [findall(bins[i] .<= e .< bins[i+1]) for i in 1:nbins]
    return groups, bins
end

function plot_conditionals(V, ε)

    groups, bins = conditional_groups(ε)

    fig = Figure(size=(700,600))
    ax = Axis(fig[1,1], title="Conditional ensemble densities")

    for (i, idx) in enumerate(groups)
        length(idx) < 5 && continue

        kd = kde((V[1,idx], V[2,idx]))

        contour!(ax, kd.x, kd.y, kd.density,
                 levels=10, linewidth=2,
                 colormap=:magma)
    end

    fig
end
V = ekrmleobj.V[end]
plot_conditionals(V,ε)

function conditional_groups_2d(ε; nbins=6)

    e1 = ε[1, :]
    e2 = ε[2, :]

    bins1 = range(minimum(e1), maximum(e1), length=nbins+1)
    bins2 = range(minimum(e2), maximum(e2), length=nbins+1)

    groups = []

    for i in 1:nbins
        for k in 1:nbins
            idx = findall(
                bins1[i] .<= e1 .< bins1[i+1] .&&
                bins2[k] .<= e2 .< bins2[k+1]
            )
            push!(groups, idx)
        end
    end

    return groups
end

V = ekrmleobj.V[end]
plot_conditionals(V,ε)


##

function make_bins_2d(ε; nbins::Int=6)
    e1 = ε[1, :]
    e2 = ε[2, :]
    bins1 = range(minimum(e1), maximum(e1), length=nbins+1)
    bins2 = range(minimum(e2), maximum(e2), length=nbins+1)
    return bins1, bins2
end

# ----------------------------
# Helper: assign each particle to a 2D bin (integer id in 1..nbins^2)
# Particles that fall on the maximum edge are nudged into the last bin.
# ----------------------------
function assign_bins_2d(ε; nbins::Int=6)
    bins1, bins2 = make_bins_2d(ε; nbins=nbins)
    e1 = ε[1, :]
    e2 = ε[2, :]

    bin_id = zeros(Int, length(e1))

    counter = 1
    for i in 1:nbins
        for j in 1:nbins
            # include the rightmost/topmost edge in the last bin to avoid zeros
            xlo, xhi = bins1[i], bins1[i+1]
            ylo, yhi = bins2[j], bins2[j+1]
            if i < nbins
                xmask = (xlo .<= e1) .& (e1 .< xhi)
            else
                xmask = (xlo .<= e1) .& (e1 .<= xhi)
            end
            if j < nbins
                ymask = (ylo .<= e2) .& (e2 .< yhi)
            else
                ymask = (ylo .<= e2) .& (e2 .<= yhi)
            end

            idx = findall(xmask .& ymask)
            bin_id[idx] .= counter
            counter += 1
        end
    end

    return bin_id, bins1, bins2
end

bin_id, bins1, bins2 = assign_bins_2d(ε; nbins=6)

# ----------------------------
# Plot: ε-space grid + v-space image
# ----------------------------
fig = Figure(size=(1000, 600))

axε = Axis(fig[1, 1], title="Noise space ε with 2D bins", xlabel="ε₁", ylabel="ε₂")
scatter!(axε, ε[1, :], ε[2, :], color=bin_id, markersize=10, colormap=:magma)

axv = Axis(fig[1, 2], title="v-space colored by ε-bin id (image of bins)", xlabel="v₁", ylabel="v₂")
scatter!(axv, V[1, :], V[2, :], color=bin_id, markersize=10, colormap=:magma)

# keep axes reasonable
tightlimits!(axε)
tightlimits!(axv)

fig

## Apply "Gaussianizing" transform

V_Gauss = zeros(2, J)
for j = 1:J
    r = norm(ℋ(V[1,j], V[2,j]) - ε[:,j])
    if r <= 1e-1
        V_Gauss[:,j] = ℋ(V[1,j], V[2,j])
    else
        continue
    end
end

fig = Figure(size=(1500, 500))


axε = Axis(fig[1, 1], title="Noise", xlabel="ε₁", ylabel="ε₂")
scatter!(axε, ε[1, :], ε[2, :], color=bin_id, markersize=10, colormap=:magma)

axv = Axis(fig[1, 2], title="Final ensemble", xlabel="v₁", ylabel="v₂")
scatter!(axv, V[1, :], V[2, :], color=bin_id, markersize=10, colormap=:magma)


axg = Axis(fig[1, 3], title="Gaussianization", xlabel="ε₁", ylabel="ε₂")
scatter!(axg, V_Gauss[1, :], V_Gauss[2, :], color=bin_id, markersize=10, colormap=:magma)


# keep axes reasonable
tightlimits!(axε)
tightlimits!(axv)
tightlimits!(axg)

display(fig)
#save("plots/FunnelGaussianizationWithTol.svg",fig)
##
C = (V_Gauss .- mean(V_Gauss; dims=2)) * (V_Gauss .- mean(V_Gauss; dims=2))' / (J-1)
C

