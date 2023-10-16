################################################################################
# Model setup
################################################################################

using nuPGCM, PyPlot, PyCall, SpecialFunctions, HDF5, Printf

# libraries
include("structs.jl")
include("plotting.jl")
include("utils.jl")
include("evolution.jl")
include("steady.jl")

# global constants
const secsInDay = 86400
const secsInYear = 360*86400
const outFolder = "out/"

"""
    m = ModelSetup(f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transportConstraint, U, Uamp, Uper)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1D(f::Float64, nz::Int64, z::Vector{Float64}, H::Float64, θ::Float64, 
                    ν_func::Function, κ_func::Function, κ_z_func::Function,
                    N2::Float64, Δt::Real, transportConstraint::Bool, U::Vector{Float64},
                    Uamp::Float64, Uper::Float64)
    # evaluate functions 
    ν = ν_func.(z)
    κ = κ_func.(z)
    κ_z = κ_z_func.(z)

    return ModelSetup1D(f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U, Uamp, Uper)
end