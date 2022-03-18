################################################################################
# Model setup
################################################################################

using PyPlot, PyCall, SpecialFunctions, HDF5, Printf, Dierckx

# libraries
include("../my_julia_lib.jl")
include("structs.jl")
include("plotting.jl")
include("utils.jl")
include("inversion.jl")
include("evolution.jl")
include("steady.jl")

# global constants
const secs_in_day = 86400
const secs_in_year = 360*86400
const out_folder = "out/"

"""
    m = ModelSetup(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transportConstraint, U, Uamp, Uper)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1DPG(bl::Bool, f::Float64, nz::Int64, z::Vector{Float64}, H::Float64, θ::Float64, 
                    ν_func::Function, κ_func::Function, κ_z_func::Function,
                    N2::Float64, Δt::Real, transportConstraint::Bool, U::Vector{Float64})
    # evaluate functions 
    ν = ν_func.(z)
    κ = κ_func.(z)
    κ_z = κ_z_func.(z)

    # pass to next funciton below
    return ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U)
end

"""
    m = ModelSetup(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1DPG(bl::Bool, f::Float64, nz::Int64, z::Vector{Float64}, H::Float64, θ::Float64, 
                    ν::Vector{Float64}, κ::Vector{Float64}, κ_z::Vector{Float64},
                    N2::Float64, Δt::Real, transportConstraint::Bool, U::Vector{Float64})
    # inversion LHS
    inversionLHS = get_inversion_LHS(ν, z, f, transportConstraint) 

    # diffusion matrix
    D = get_diffusion_matrix(z, κ)

    # return struct
    return ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, inversionLHS, D, transportConstraint, U)
end
