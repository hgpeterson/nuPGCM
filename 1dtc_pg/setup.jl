################################################################################
# Model setup
################################################################################

using PyPlot, PyCall, SpecialFunctions, HDF5, Printf

# libraries
include("../myJuliaLib.jl")
include("structs.jl")
include("plotting.jl")
include("utils.jl")
include("inversion.jl")
include("evolution.jl")
include("steady.jl")

# global constants
const secsInDay = 86400
const secsInYear = 360*86400
const outFolder = "out/"

"""
    m = ModelSetup(f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transportConstraint, U₀)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1DPG(f::Float64, nz::Int64, z::Array{Float64,1}, H::Float64, θ::Float64, 
                    ν_func::Function, κ_func::Function, κ_z_func::Function,
                    N2::Float64, Δt::Real, transportConstraint::Bool, U₀::Float64)
    # evaluate functions 
    ν = ν_func.(z)
    κ = κ_func.(z)
    κ_z = κ_z_func.(z)

    # pass to next funciton below
    return ModelSetup1DPG(f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U₀)
end

"""
    m = ModelSetup(f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U₀)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1DPG(f::Float64, nz::Int64, z::Array{Float64,1}, H::Float64, θ::Float64, 
                    ν::Array{Float64,1}, κ::Array{Float64,1}, κ_z::Array{Float64,1},
                    N2::Float64, Δt::Real, transportConstraint::Bool, U₀::Float64)
    # inversion LHS
    inversionLHS = getInversionLHS(ν, z, f, θ, transportConstraint) 

    # diffusion matrix
    D = getDiffusionMatrix(z, κ)

    # return struct
    return ModelSetup1DPG(f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, inversionLHS, D, transportConstraint, U₀)
end
