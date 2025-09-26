using Test
using nuPGCM.FiniteElements
using Printf

# constructed solution and corresponding RHS
u₀(x) = sin(π*x[1]) * cos(π*x[2])
f(x) = 2π^2 * sin(π*x[1]) * cos(π*x[2])

n = 100
mesh = Mesh(n, n)
fe_data = FEData(mesh; quad_deg=4)
space = P2()
dof_data = DoFData(mesh, space)
K = FiniteElements.stiffness_matrix(mesh, fe_data.jacobians, fe_data.quad_rule, space, dof_data; dirichlet=["boundary"])
rhs = FiniteElements.rhs_vector(mesh, fe_data.jacobians, fe_data.quad_rule, space, dof_data, f, u₀; dirichlet=["boundary"])
u = FEField(fe_data, space, dof_data)
u.values .= K\rhs

e = L2_error(u, u₀)
@printf("L2 error: %.2e\n", e)
@test e < 1e-6
e = maximum(abs(u.values[i] - u₀(mesh.nodes[i, :])) for i in axes(mesh.nodes, 1))
@printf("Max error at nodes: %.2e\n", e)
@test e < 1e-6

FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))