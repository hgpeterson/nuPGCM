using Test
using nuPGCM.FiniteElements
using Printf

# constructed solution and corresponding RHS
u₀(x) = sin(π*x[1]) * cos(π*x[2])
v₀(x) = cos(π*x[1]) * sin(π*x[2])
fˣ(x) = 6π^2 * sin(π*x[1]) * cos(π*x[2])
fʸ(x) = 6π^2 * cos(π*x[1]) * sin(π*x[2])
# u₀(x) = 0
# v₀(x) = 0
# function fˣ(x)
#     if ((x[1] - 0.25)^2 + (x[2] - 0.5)^2 < 0.1^2) || ((x[1] - 0.75)^2 + (x[2] - 0.5)^2 < 0.1^2)
#         return 1
#     else
#         return 0
#     end
# end
# function fʸ(x)
#     if (x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.1^2
#         return 1
#     else
#         return 0
#     end
# end

n = 100
mesh = Mesh(n, n)
fe_data = FEData(mesh; quad_deg=4)
space = P2()
dof_data = DoFData(mesh, space)
@info "DoFs: $(dof_data.N)"
dirichlet = ["left", "bottom", "right", "top"]
@time "build K" K = FiniteElements.elasticity_matrix(fe_data, space, dof_data; dirichlet, d=2)
rhsˣ = FiniteElements.rhs_vector(fe_data, space, dof_data, fˣ, u₀; dirichlet)
rhsʸ = FiniteElements.rhs_vector(fe_data, space, dof_data, fʸ, v₀; dirichlet)
rhs = [rhsˣ; rhsʸ]
u = FEField(fe_data, space, dof_data)
v = FEField(fe_data, space, dof_data)
sol = K\rhs
u.values .= sol[1:dof_data.N]
v.values .= sol[dof_data.N+1:end]

e = L2_error(u, u₀) + L2_error(v, v₀)
@printf("L2 error: %.2e\n", e)
# @test e < 1e-6

FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u, 
                                                                     "v" => v, 
                                                                     "u₀" => u₀, 
                                                                     "v₀" => v₀, 
                                                                     "fˣ" => fˣ, 
                                                                     "fʸ" => fʸ))