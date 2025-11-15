export heat_eq_params, solve_HE


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