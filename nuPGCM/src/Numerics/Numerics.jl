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
    tri_area,
    local_basis_func

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