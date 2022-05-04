module Numerics

export
    # derivatives
    mkfdstencil,
    differentiate_pointwise,
    differentiate,

    # integrals
    trapz,
    cumtrapz,
    gaussian_quad2,

    # finite elements
    load_mesh,
    tri_area,
    get_linear_basis_coeffs,
    local_basis_func,
    evaluate

using PyPlot
using PyCall
using SpecialFunctions
using Printf
using SparseArrays
using SuiteSparse
using LinearAlgebra
using HDF5

include("derivatives.jl")
include("integrals.jl")
include("finite_elements.jl")

end # module