using Test
using nuPGCM.FiniteElements

# mesh = FiniteElements.Mesh(joinpath(@__DIR__, "../meshes/bowl2D_1.000000e-02_5.000000e-01.msh"))
h = 0.001
nodes = [0.0 0.0 0.0
         0.0 1.0 0.0
         1.0 0.0 0.0
         1.0 1.0 0.0]
elements = [1 2 4
            1 3 4]
boundary_nodes = Dict("left" => [1, 2], 
                      "right" => [3, 4], 
                      "bottom" => [1, 3], 
                      "top" => [2, 4])
mesh = FiniteElements.Mesh(nodes, elements, boundary_nodes)


jacs = Jacobians(mesh)
quad = QuadratureRule(Triangle(); deg=2)
space = P1()
# space = Mini()
M = FiniteElements.mass_matrix(mesh, jacs, quad, space; dirichlet=["bottom"])
K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space; dirichlet=["bottom"])
rhs = M*ones(size(M, 1))
rhs[FiniteElements.get_dirichlet_tags(mesh, ["bottom"])] .= 0.0
u = K\rhs  #  FIXME: not getting -y^2/2 + y like I thought
FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))