module nuPGCM

export 
    # constants
    secs_in_day,
    secs_in_year,
    out_folder,
    set_out_folder,

    ### Numerics ###

    # derivatives
    mkfdstencil,
    differentiate_pointwise,
    differentiate,

    # integrals
    trapz,
    cumtrapz,
    gaussian_quad2,
    tri_quad,

    # finite elements
    ShapeFunctions,
    ShapeFunctionIntegrals,
    Grid,
    Jacobians,
    L2norm,
    H1norm,
    all_edges,
    add_midpoints,
    get_t_dict,
    tri_area,
    transform_from_std_tri,
    transform_to_std_tri,
    get_shape_func_coeffs,
    shape_func,
    fem_evaluate,

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
    ModelSetup1DPG,
    ModelState1DPG,

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
    ModelSetup2DPG,
    ModelState2DPG,

    # plotting
    ridge_plot,
    plot_state_2DPG,

    # BL theory
    get_full_soln,

    ### 3D Model ###

    # loading
    load_setup_3D,
    load_state_3D,

    # model structs 
    ModelSetup3DPG,
    ModelState3DPG,

    # inversion
    get_barotropic_LHS,
    get_barotropic_RHS,
    get_baroclinic_LHS,
    get_baroclinic_RHS,
    solve_baroclinic_systems,
    get_τ_b,
    get_full_τ,
    get_u,

    # plotting
    tplot,
    plot_horizontal,
    plot_ξ_slice,
    plot_η_slice,

    # operators
    ∂ξ,
    ∂η,
    curl

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

# global constants
const secs_in_day = 86400
const secs_in_year = 360*86400

# default output folder, can be changed
out_folder = ""
function set_out_folder(of::String)
    global out_folder = of
end

# include code
include("Numerics/derivatives.jl")
include("Numerics/integrals.jl")
include("Numerics/finite_elements.jl")

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
include("ThreeDimensionalModel/operators.jl")
include("ThreeDimensionalModel/inversion.jl")
include("ThreeDimensionalModel/evolution.jl")
include("ThreeDimensionalModel/plotting.jl")
include("ThreeDimensionalModel/boundary_layer.jl")

end # module
