using OralQual
using LinearAlgebra
using Random
using CairoMakie
using Distributions


@inline colmean(V::AbstractMatrix) = vec(mean(V, dims=2))

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

## Setup
n = 50
d = 5
J = 10000
prob = randomLinearProblemObj(n, d, J)


## EKRMLE
obj = EKRMLEObj(prob.V0, prob.y, prob.Γ) # create EKRMLE object
iters = 100
H_s(prob::randomLinearProblemObj, v::AbstractVector) = prob.H * v
EKRMLE_run!(obj, prob, H_s, iters)

##
C = _samplecov(obj.V[end])
## 
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1])
ax1.yreversed=true
ax2 = Axis(fig[1, 3])
ax2.yreversed=true
hm1 = heatmap!(ax1, C; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, prob.pHess; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)

## Spectral projectors
projectors = spectralproj(prob, _samplecov(prob.V0))
P = real.(projectors.P)

## 
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1])
ax1.yreversed=true
ax2 = Axis(fig[1, 3])
ax2.yreversed=true
hm1 = heatmap!(ax1, P*C*P'; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, P*prob.pHess*P'; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)

## plot some marginals
m1 = 1
m2 = 2
V_marg = obj.V[end][[m1,m2],:]
PV_marg = (P*obj.V[end])[[m1,m2],:]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=15, label="true")
scatter!(ax, PV_marg[1,:], PV_marg[2,:]; markersize=15, marker=:cross ,label = "P-space")

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)


## 
μ = colmean(obj.V[end])
Pμ = P*μ
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title = "Full space")
ax2 = Axis(fig[2,1], title = "P space")
scatterlines!(ax1, prob.v_star;linewidth=3, markersize=15, label="LS")
scatterlines!(ax1, μ;linewidth=3, markersize=15, label="μ")
scatterlines!(ax2, P*prob.v_star;linewidth=5, markersize=20, label="P-space LS")
scatterlines!(ax2, Pμ;linestyle=:dash, linewidth=4, marker=:cross, markersize=18, label="P-space")


display(fig)

## compare against black box sampler
bb_ens = rand(MvNormal(P*prob.v_star, Symmetric(P*prob.pHess*P'+1e-15*I)), J)

m1 = 1
m2 = 2
bb_marg = bb_ens[[m1,m2],:]
PV_marg = (P*obj.V[end])[[m1,m2],:]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="RMLE in P-space")
scatter!(ax, bb_marg[1,:], bb_marg[2,:]; markersize=15, label="black-box")
scatter!(ax, PV_marg[1,:], PV_marg[2,:]; markersize=15, marker=:cross ,label = "EKRMLE")

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)