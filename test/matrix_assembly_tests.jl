using Test
using nuPGCM.FiniteElements

# constructed solution and corresponding RHS
u₀(x) = sin(π*x[1]) * cos(π*x[2])
f(x) = 2π^2 * sin(π*x[1]) * cos(π*x[2])

function solve(mesh::Mesh)
    jacs = Jacobians(mesh)
    quad = QuadratureRule(FiniteElements.get_element_type(mesh); deg=2)
    space = P1()
    dof_data = DoFData(mesh, space)
    K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space, dof_data; dirichlet=["boundary"])
    rhs = FiniteElements.rhs_vector(mesh, jacs, quad, space, dof_data, f, u₀; dirichlet=["boundary"])
    u = K\rhs
    e = abs.(u .- u₀.(eachrow(mesh.nodes)))
    println("Max error: $(maximum(e))")
    FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u, "e" => e))
    return u
end

# nodes = [0.0 0.0 0.0
#          1.0 0.0 0.0
#          0.0 1.0 0.0]
# elements = [1 2 3]
# boundary_nodes = Dict("boundary" => [1, 2, 3]) 
# mesh = FiniteElements.Mesh(nodes, elements, boundary_nodes)
# jacs = Jacobians(mesh)
# quad = QuadratureRule(Triangle(); deg=2)
# space = P1()
# M = FiniteElements.mass_matrix(mesh, jacs, quad, space)
# @test M ≈ [1/12 1/24 1/24; 1/24 1/12 1/24; 1/24 1/24 1/12]

function square_mesh(nx, ny)
    x = range(0.0, 1.0; length=nx)
    y = range(0.0, 1.0; length=ny)
    nodes = zeros(nx*ny, 3)
    for j in 1:ny, i in 1:nx
        nodes[(j-1)*nx + i, :] = [x[i], y[j], 0.0]
    end
    elements = zeros(Int, (nx-1)*(ny-1)*2, 3)
    k = 1
    for j in 1:(ny-1), i in 1:(nx-1)
        n1 = (j-1)*nx + i
        n2 = (j-1)*nx + i + 1
        n3 = j*nx + i
        n4 = j*nx + i + 1
        elements[k, :] = [n1, n2, n4]
        k += 1
        elements[k, :] = [n1, n4, n3]
        k += 1
    end
    boundary_nodes = Dict("boundary" => unique(vcat(1:nx, (nx-1)*nx .+ (1:nx), 1:nx:(nx*(ny-1)+1), nx:nx:(nx*ny))))
    return FiniteElements.Mesh(nodes, elements, boundary_nodes)
end
mesh = square_mesh(100, 100)
jacs = Jacobians(mesh)
quad = QuadratureRule(FiniteElements.get_element_type(mesh); deg=2)
space = P1()
dof_data = DoFData(mesh, space)
K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space, dof_data; dirichlet=["boundary"])
rhs = FiniteElements.rhs_vector(mesh, jacs, quad, space, dof_data, f, u₀; dirichlet=["boundary"])
u = K\rhs
n_nodes = size(mesh.nodes, 1)
e = abs.(u[1:n_nodes] .- u₀.(eachrow(mesh.nodes)))  # FIXME: convergence doesn't look good
println("Max error: $(maximum(e))")
# FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u, "e" => e))
# FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))
# FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), u)
# u = solve(mesh)

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