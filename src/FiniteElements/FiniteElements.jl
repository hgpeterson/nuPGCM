module FiniteElements

using Gmsh: gmsh
using LinearAlgebra
using SparseArrays
using WriteVTK
using ProgressMeter

export 
    # elements.jl
    Point,
    Line,
    Triangle,
    Tetrahedron,

    # mesh.jl
    Mesh,

    # quadrature.jl
    Jacobians,
    QuadratureRule,
    integrate,
    ∫,

    # shape_functions.jl
    Lagrange,
    φ,
    ∇φ,

    # vtk.jl
    save_vtu

include("elements.jl")
include("mesh.jl")
include("quadrature.jl")
include("shape_functions.jl")
include("matrix_assembly.jl")
include("vtu.jl")

end # module