module ThreeDimensionalModel

export 
    # constants
    secs_in_day,
    secs_in_year,
    out_folder,

    # saving and loading
    save_setup_3DPG,
    load_setup_3DPG,
    save_state_3DPG,
    load_state_3DPG,

    # model structs 
    ModelSetup3DPG,
    ModelState3DPG,

    # inversion
    get_barotropic_LHS,
    get_barotropic_RHS,

    # plotting
    tplot,
    plot_horizontal

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

include("setup.jl")
include("logging.jl")
include("operators.jl")
include("inversion.jl")
include("evolution.jl")
include("plotting.jl")
include("boundary_layer.jl")

end # module