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
K = FiniteElements.elasticity_matrix(mesh, fe_data.jacobians, fe_data.quad_rule, space, dof_data; dirichlet=["boundary"], d=2)
rhs = FiniteElements.rhs_vector(mesh, fe_data.jacobians, fe_data.quad_rule, space, dof_data, x->1, x->0; dirichlet=["boundary"])
n = length(rhs)
rhs = [rhs; zeros(n)]
u = FEField(fe_data, space, dof_data)
v = FEField(fe_data, space, dof_data)
sol = K\rhs
u.values .= sol[1:n]
v.values .= sol[n+1:end]

# e = L2_error(u, u₀)
# @printf("L2 error: %.2e\n", e)
# @test e < 1e-6
# e = maximum(abs(u.values[i] - u₀(mesh.nodes[i, :])) for i in axes(mesh.nodes, 1))
# @printf("Max error at nodes: %.2e\n", e)
# @test e < 1e-6

FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u, "v" => v))