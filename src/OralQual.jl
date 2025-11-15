module OralQual


using LinearAlgebra
using Statistics
using Distributions
using Random
using CairoMakie
using BlockDiagonals
using LaTeXStrings
using ColorSchemes
using SparseArrays
using MAT
using ControlSystems



include("EKRMLE.jl")
include("DarcyHelpers1D.jl")
include("DarcyHelpers2D.jl")
include("LinearHelpers.jl")
include("HeatEqHelpers.jl")



end # module OralQual
