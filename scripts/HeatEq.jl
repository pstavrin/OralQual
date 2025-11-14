using OralQual
using LinearAlgebra
using CairoMakie
using Distributions
using Random
using ColorSchemes
using MAT
using SparseArrays


mutable struct heat_eq_params{T<:AbstractFloat, TI<:Int, UI<:UInt8}
    # Problem dimensions
    n::TI # observation space dimension
    d::TI # state space dimension
    
    # Solver params
    Δt::T # time step
    T_stop::T # end time
    TT::Vector{T} # time domain
    h::T # time between measurements
    T_meas::Vector{T} # measurement times

    # LTI Operators
    A::SparseArrays.AbstractSparseMatrixCSC{T,TI}
    C::SparseArrays.AbstractSparseMatrixCSC{UI,TI}

    # Forward opertor
    H::AbstractMatrix{T}

end

function heat_eq_params(
    n::TI,
    Δt::T,
    T_stop::T,
    path::AbstractString
) where {T<:AbstractFloat, TI<:Int}
    operators = get_heat_data(path)
    d = size(operators.A, 2)
    TT = collect(zero(T):Δt:T_stop)
    h = T_stop/n
    T_meas = collect(h:h:T_stop)

    # construct H explicitly
    A, C = operators.A, operators.C
    skips = Int(round(h/Δt))
    M = I + Δt*A
    Mks = M^skips
    H = zeros(T,n,d)
    Mpow = I
    for i = 1 : n
        Mpow = Mks*Mpow
        H[i,:] = vec(C*Mpow)'
    end


    heat_eq_params(n,d,Δt,T_stop,TT,h,T_meas,A,C,H)
    
end

function get_heat_data(path::AbstractString)
    data = matread(path)
    A, C = data["A"], data["C"]
    return (A = A, C = C)
end

## 
function solve_HE(heat::heat_eq_params{T,TI,UI}, v::AbstractVector{T}) where {T<:AbstractFloat, TI<:Int, UI<:UInt8}
    N_T = length(heat.TT)
    A, C = heat.A, heat.C
    Δt = heat.Δt
    y = zeros(T, N_T, 1)
    y[1] = (C*v)[1]
    for i = 2 : N_T
        v .+= Δt*(A*v)
        y[i] = (C*v)[1]
    end
    return y
    
end






## Setup
path = "data/heat-cont.mat"
n = 100
Δt = 1e-3
T_stop = 10.0
heat = heat_eq_params(n, Δt, T_stop, path)
d = heat.d

## noisy data
truth = rand(Normal(0, 1), d)
sol_nonoise = solve_HE(heat, truth)
h, Δt = heat.h, heat.Δt
meas_idx = Int.(round.(heat.T_meas ./Δt))
y_nonoise = sol_nonoise[meas_idx]
σ = 0.5*maximum(y_nonoise)
γ = fill(σ^2, n)
Γ = γ .* I(n)
ε = rand(MvNormal(vec(zeros(1,n)),Γ))
y = y_nonoise + ε

## Plot data
fig = Figure()
ax = Axis(fig[1,1],
    xlabel=L"Index",
)

scatter!(ax, heat.TT, vec(sol_nonoise), label= L"\text{solution}")
scatter!(ax, heat.T_meas, y, marker=:cross, label = L"y")
axislegend(position =:lb, labelsize = 20)
display(fig)

