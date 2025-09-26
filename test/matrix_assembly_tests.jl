using Test
using nuPGCM.FiniteElements
using Printf

# constructed solution and corresponding RHS
u₀(x) = sin(π*x[1]) * cos(π*x[2])
f(x) = 2π^2 * sin(π*x[1]) * cos(π*x[2])

function square_mesh(nx, ny)
    x = range(0.0, 1.0; length=nx)
    y = range(0.0, 1.0; length=ny)
    nodes = zeros(nx*ny, 3)
    imap = reshape(1:(nx*ny), nx, ny)
    for i in 1:nx, j in 1:ny
        nodes[imap[i, j], :] = [x[i], y[j], 0.0]
    end
    elements = zeros(Int, 2*(nx-1)*(ny-1), 3)
    k = 1
    for i in 1:(nx-1), j in 1:(ny-1)
        n1 = imap[i, j]
        n2 = imap[i, j+1]
        n3 = imap[i+1, j]
        n4 = imap[i+1, j+1]
        elements[k, :] = [n1, n4, n2]
        k += 1
        elements[k, :] = [n1, n3, n4]
        k += 1
    end
    boundary_nodes = Dict("boundary" => unique(vcat(imap[1, :], imap[end, :], imap[:, 1], imap[:, end])))
    return FiniteElements.Mesh(nodes, elements, boundary_nodes)
end

n = 100
mesh = square_mesh(n, n)
jacs = Jacobians(mesh)
quad = QuadratureRule(FiniteElements.get_element_type(mesh); deg=4)
space = P2()
dof_data = DoFData(mesh, space)
K = FiniteElements.elasticity_matrix(mesh, jacs, quad, space, dof_data; dirichlet=["boundary"], d=2)
rhs = FiniteElements.rhs_vector(mesh, jacs, quad, space, dof_data, x->1, x->0; dirichlet=["boundary"])
n = length(rhs)
rhs = [rhs; zeros(n)]
u = K\rhs
FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u[1:n], "v" => u[n+1:2n]))
# K = FiniteElements.stiffness_matrix(mesh, jacs, quad, space, dof_data; dirichlet=["boundary"])
# rhs = FiniteElements.rhs_vector(mesh, jacs, quad, space, dof_data, f, u₀; dirichlet=["boundary"])
# u = K\rhs
function compute_L2_error(u)
    # u at quadrature points
    el = FiniteElements.get_element_type(mesh)
    uq = zeros(eltype(u), size(mesh.elements, 1), length(quad.weights))
    for k in 1:size(mesh.elements, 1)
        for q in eachindex(quad.weights)
            φq = φ(el, space, quad.points[q, :])
            for i in 1:FiniteElements.n_dofs(el, space)
                uq[k, q] += u[dof_data.global_dof[k, i]] * φq[i]
            end
        end
    end

    # integrate
    e = zero(eltype(u))
    for k in 1:size(mesh.elements, 1)
        for q in eachindex(quad.weights)
            x = FiniteElements.transform_from_reference(el, jacs.∂x∂ξ[k, :, :], quad.points[q, :], mesh.nodes[mesh.elements[k, :], :])
            e += quad.weights[q] * (uq[k, q] - u₀(x))^2 * jacs.dV[k]
        end
    end

    return √e
end
# e = compute_L2_error(u)
# @printf("L2 error: %.2e\n", e)
# e = maximum(abs(u[i] - u₀(mesh.nodes[i, :])) for i in axes(mesh.nodes, 1))
# @printf("Max error at nodes: %.2e\n", e)
# FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), Dict("u" => u))
# FiniteElements.save_vtu(mesh, joinpath(@__DIR__, "data/u.vtu"), u)