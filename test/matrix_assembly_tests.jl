using Test
using nuPGCM.FiniteElements

# constructed solution and corresponding RHS
u₀(x) = sin(π*x[1]) * cos(π*x[2])
f(x) = 2π^2 * sin(π*x[1]) * cos(π*x[2])

# nodes = [0.0 0.0 0.0
#          1.0 0.0 0.0
#          0.0 1.0 0.0]
# elements = [1 2 3]
# boundary_nodes = Dict("boundary" => [1, 2, 3]) 
# mesh = FiniteElements.Mesh(nodes, elements, boundary_nodes)
# # mesh.edges .= [1 2; 2 3; 3 1]
# # mesh.emap .= [1 2 3]
# jacs = Jacobians(mesh)
# quad = QuadratureRule(Triangle(); deg=4)
# space = P2()
# dof_data = DoFData(mesh, space)
# M = FiniteElements.mass_matrix(mesh, jacs, quad, space, dof_data)
# # @test M ≈ [1/12 1/24 1/24; 1/24 1/12 1/24; 1/24 1/24 1/12]
# @test M ≈ [1/60 -1/360 -1/360 0 -1/90 0; -1/360 1/60 -1/360 0 0 -1/90; -1/360 -1/360 1/60 -1/90 0 0; 0 0 -1/90 4/45 2/45 2/45; -1/90 0 0 2/45 4/45 2/45; 0 -1/90 0 2/45 2/45 4/45]
# # φ1(ξ) = φ(Triangle(), space, ξ)[1]
# # φ6(ξ) = φ(Triangle(), space, ξ)[6]
# K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space, dof_data)
# # @test K ≈ [1 -1/2 -1/2; -1/2 1/2 0; -1/2 0 1/2]
# @test K ≈ [1 1/6 1/6 -(2/3) 0 -(2/3); 1/6 1/2 0 -(2/3) 0 0; 1/6 0 1/2 0 0 -(2/3); -(2/3) -(2/3) 0 8/3 -(4/3) 0; 0 0 0 -(4/3) 8/3 -(4/3); -(2/3) 0 -(2/3) 0 -(4/3) 8/3]

# function square_mesh(nx, ny)
#     x = range(0.0, 1.0; length=nx)
#     y = range(0.0, 1.0; length=ny)
#     nodes = zeros(nx*ny, 3)
#     for j in 1:ny, i in 1:nx
#         nodes[(j-1)*nx + i, :] = [x[i], y[j], 0.0]
#     end
#     elements = zeros(Int, (nx-1)*(ny-1)*2, 3)
#     k = 1
#     for j in 1:(ny-1), i in 1:(nx-1)
#         n1 = (j-1)*nx + i
#         n2 = (j-1)*nx + i + 1
#         n3 = j*nx + i
#         n4 = j*nx + i + 1
#         elements[k, :] = [n1, n2, n4]
#         k += 1
#         elements[k, :] = [n1, n4, n3]
#         k += 1
#     end
#     boundary_nodes = Dict("boundary" => unique(vcat(1:nx, (nx-1)*nx .+ (1:nx), 1:nx:(nx*(ny-1)+1), nx:nx:(nx*ny))))
#     return FiniteElements.Mesh(nodes, elements, boundary_nodes)
# end
# mesh = square_mesh(100, 100)
# jacs = Jacobians(mesh)
# quad = QuadratureRule(FiniteElements.get_element_type(mesh); deg=2)
# space = P2()
# dof_data = DoFData(mesh, space)
# K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space, dof_data; dirichlet=["boundary"])
# rhs = FiniteElements.rhs_vector(mesh, jacs, quad, space, dof_data, f, u₀; dirichlet=["boundary"])
# u = K\rhs
# function compute_L2_error(u)
#     # u at quadrature points
#     el = FiniteElements.get_element_type(mesh)
#     uq = zeros(eltype(u), size(mesh.elements, 1), length(quad.weights))
#     for k in 1:size(mesh.elements, 1)
#         for q in eachindex(quad.weights)
#             φq = φ(el, space, quad.points[q, :])
#             for i in 1:FiniteElements.n_dofs(el, space)
#                 uq[k, q] += u[dof_data.global_dof[k, i]] * φq[i]
#             end
#         end
#     end

#     # integrate
#     e = zero(eltype(u))
#     for k in 1:size(mesh.elements, 1)
#         for q in eachindex(quad.weights)
#             x = FiniteElements.transform_from_reference(el, jacs.∂x∂ξ[k], quad.points[q, :], mesh.nodes[mesh.elements[k, :], :])
#             e += quad.weights[q] * (uq[k, q] - u₀(x))^2 * jacs.dV[k]
#         end
#     end

#     return √e
# end
# e = compute_L2_error(u)
# println("Max error: $(maximum(e))")
# # P1, 10:  0.06624248766480319
# # P1, 100: 0.005859794296430321
# # P2, 10:  0.0710491994838959
# # P2, 100: 0.006477370892311304
# # FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u, "e" => e))
FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))
# # FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), u)

# nodes = [0.0 0.0 0.0
#          0.0 1.0 0.0
#          1.0 0.0 0.0
#          1.0 1.0 0.0]
# elements = [1 2 4
#             1 3 4]
# boundary_nodes = Dict("left" => [1, 2], 
#                       "right" => [3, 4], 
#                       "bottom" => [1, 3], 
#                       "top" => [2, 4])
# mesh = FiniteElements.Mesh(nodes, elements, boundary_nodes)
# jacs = Jacobians(mesh)
# quad = QuadratureRule(Triangle(); deg=2)
# space = P1()
# # space = Mini()
# M = FiniteElements.mass_matrix(mesh, jacs, quad, space; dirichlet=["bottom"])
# K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space; dirichlet=["bottom"])
# rhs = M*ones(size(M, 1))
# rhs[FiniteElements.get_dirichlet_tags(mesh, ["bottom"])] .= 0.0
# u = K\rhs
# FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))

# mesh = FiniteElements.Mesh(joinpath(@__DIR__, "../meshes/bowl2D_1.000000e-02_5.000000e-01.msh"))