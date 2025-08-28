using Test
using nuPGCM.FiniteElements

mesh = FiniteElements.Mesh(joinpath(@__DIR__, "../meshes/bowl2D_1.000000e-02_5.000000e-01.msh"))
jacs = Jacobians(mesh)
quad = QuadratureRule(Triangle(); deg=2)
space = Mini()
M = FiniteElements.mass_matrix(mesh, jacs, quad, space; dirichlet_tags=["bot"])
K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space; dirichlet_tags=["bot"])
rhs = M*ones(size(M, 1))
rhs[FiniteElements.get_dirichlet_tags(mesh, ["bot"])] .= 1.0
u = K\rhs
FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))