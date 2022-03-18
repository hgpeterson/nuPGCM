module OneDimensionalModel

export 
    # constants
    secs_in_day,
    secs_in_year,
    out_folder,

    # saving and loading
    save_setup_1DPG,
    load_setup_1DPG,
    save_state_1DPG,
    load_state_1DPG,

    # model structs 
    ModelSetup1DPG,
    ModelState1DPG,

    # inversion
    invert,
    invert!,

    # evolution
    evolve!,

    # plotting
    profile_plot,

    # BL theory
    get_BL_correction,
    get_BL_params,
    get_full_soln,

    # steady state (only for canonical 1D model)
    get_steady_state

using nuPGCM
using nuPGCM.Numerics

using PyPlot
using PyCall
using SpecialFunctions
using Printf
using SparseArrays
using SuiteSparse
using LinearAlgebra
using HDF5
using Dierckx

include("setup.jl")
include("logging.jl")
include("operators.jl")
include("inversion.jl")
include("evolution.jl")
include("plotting.jl")
include("boundary_layer.jl")
include("steady.jl")

end # module