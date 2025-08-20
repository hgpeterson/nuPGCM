module FiniteElements

using Gmsh: gmsh
using LinearAlgebra
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
    ∫

include("elements.jl")
include("mesh.jl")
include("quadrature.jl")

end # module