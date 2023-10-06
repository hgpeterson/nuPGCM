module nuPGCM

export 
    # constants
    secs_in_day,
    secs_in_year,
    out_folder,
    set_out_folder,

    ### Numerics ###

    # interpolation
    lerp,

    # derivatives
    mkfdstencil,
    differentiate_pointwise,
    differentiate,

    # integrals
    trapz,
    cumtrapz,
    quad_weights_points,

    # finite elements
    AbstractElement,
    Line,
    Triangle,
    Wedge,
    φ,
    ∂φ,
    φξ,
    φη,
    φζ,
    transform_from_ref_el,
    transform_to_ref_el,
    Grid,
    add_midpoints,
    all_edges,
    Jacobians,
    AbstractField,
    FEField,
    DGField,
    FVField,
    L2norm,
    transform_from_ref_el,
    transform_to_ref_el,
    ∂,
    ∂x,
    ∂y,
    ∂z,

    ### All models ###

    # saving models
    save_setup,
    save_state,

    # inversion
    invert,
    invert!,

    # evolution
    evolve!,

    # plotting
    profile_plot,

    ### 1D Model ### 

    # loading
    load_setup_1D,
    load_state_1D,

    # model structs 
    ModelSetup1D,
    ModelState1D,

    # BL theory
    get_BL_correction,
    get_BL_params,
    get_full_soln,

    # steady state (only for canonical 1D model)
    get_steady_state,

    ### 2D Model ###

    # operators
    ∂ξ,
    ∂σ,
    ∂x,
    ∂z,
    transform_from_TF,

    # loading
    load_setup_2D,
    load_state_2D,

    # model structs 
    ModelSetup2D,
    ModelState2D,

    # plotting
    ridge_plot,
    plot_state_2DPG,

    # BL theory
    get_full_soln,

    ### 3D Model ###

    # model structs
    ModelSetup3D,
    ModelState3D,

    # plotting
    plot_profiles,
    plot_xslices,
    plot_yslices,

    # loading
    load_setup_3D,
    load_state_3D

# packages
using PyPlot
using PyCall
using SpecialFunctions
using Printf
using SparseArrays
using SuiteSparse
using LinearAlgebra
using Dierckx
using HDF5
using ProgressMeter
using WriteVTK
using Delaunay
using IterativeSolvers
using CUDA

# global constants
const secs_in_day = 86400.
const secs_in_year = 360*86400.

# python imports (https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules)
const mpl = PyNULL()
function __init__()
    copy!(mpl, pyimport("matplotlib"))
end

# default output folder, can be changed
out_folder = "output"
function set_out_folder(of::String)
    global out_folder = of
end

# include code
include("Numerics/interpolation.jl")
include("Numerics/derivatives.jl")
include("Numerics/elements.jl")
include("Numerics/integrals.jl")
include("Numerics/grids.jl")
include("Numerics/fields.jl")

include("OneDimensionalModel/setup.jl")
include("OneDimensionalModel/logging.jl")
include("OneDimensionalModel/operators.jl")
include("OneDimensionalModel/inversion.jl")
include("OneDimensionalModel/evolution.jl")
include("OneDimensionalModel/plotting.jl")
include("OneDimensionalModel/boundary_layer.jl")
include("OneDimensionalModel/steady.jl")

include("TwoDimensionalModel/setup.jl")
include("TwoDimensionalModel/logging.jl")
include("TwoDimensionalModel/operators.jl")
include("TwoDimensionalModel/inversion.jl")
include("TwoDimensionalModel/evolution.jl")
include("TwoDimensionalModel/plotting.jl")
include("TwoDimensionalModel/boundary_layer.jl")

include("ThreeDimensionalModel/setup.jl")
include("ThreeDimensionalModel/logging.jl")
include("ThreeDimensionalModel/mesh_utils.jl")
include("ThreeDimensionalModel/baroclinic.jl")
include("ThreeDimensionalModel/barotropic.jl")
include("ThreeDimensionalModel/inversion.jl")
include("ThreeDimensionalModel/evolution.jl")
include("ThreeDimensionalModel/plotting.jl")
include("ThreeDimensionalModel/boundary_layer.jl")
include("ThreeDimensionalModel/energy.jl")

end # module
