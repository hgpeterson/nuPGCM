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

    # spaces.jl
    P1,
    P2,
    Bubble,
    Mini,
    φ,
    ∇φ,

    # dofs.jl
    DoFData,

    # vtk.jl
    save_vtu

include("elements.jl")
include("mesh.jl")
include("quadrature.jl")
include("spaces.jl")
include("dofs.jl")
include("matrix_assembly.jl")
include("vtu.jl")

end # module