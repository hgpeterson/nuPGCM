struct SparseIJV{VI, VF}
    I::VI
    J::VI
    V::VF
end

function SparseIJV(T)
    return SparseIJV(Int[], Int[], T[])
end

function sparse_csc(s::SparseIJV, m, n)
    return sparse(s.I, s.J, s.V, m, n)
end

function stamp!(global_matrix::SparseIJV, local_matrix, element_nodes, i_diri)
    for i_local in axes(local_matrix, 1)
        i_global = element_nodes[i_local]
        if i_global in i_diri
            continue
        end
        for j_local in axes(local_matrix, 2)
            j_global = element_nodes[j_local]
            push!(global_matrix.I, i_global)
            push!(global_matrix.J, j_global)
            push!(global_matrix.V, local_matrix[i_local, j_local])
        end
    end
    return global_matrix
end

function add_dirichlet!(global_matrix::SparseIJV, i_diri)
    T = eltype(global_matrix.V)
    for i in i_diri
        push!(global_matrix.I, i)
        push!(global_matrix.J, i)
        push!(global_matrix.V, one(T))  # Dirichlet condition
    end
    return global_matrix
end

##

function mass_matrix(mesh::Mesh, jacs::Jacobians, quad::QuadratureRule, space::AbstractFESpace;
                     dirichlet_tags=String[])
    el = mesh.element_type
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    i_diri = get_dirichlet_tags(mesh, dirichlet_tags)

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    global_dof = get_global_dof(mesh, space)
    local_matrix = zeros(T, n_dof_per_el, n_dof_per_el)
    for k in 1:n_el
        local_matrix .= zero(T)
        for q in eachindex(quad.weights)
            φq = φ(el, space, quad.points[q, :])
            for i in 1:n_dof_per_el, j in 1:n_dof_per_el
                local_matrix[i, j] += quad.weights[q] * φq[i] * φq[j] * jacs.dV[k]
            end
        end
        stamp!(global_matrix, local_matrix, global_dof[k, :], i_diri)
    end

    add_dirichlet!(global_matrix, i_diri)

    n_nodes = size(mesh.nodes, 1)
    return sparse_csc(global_matrix, n_nodes, n_nodes)
end

function stiffness_matrix(mesh::Mesh, jacs::Jacobians, quad::QuadratureRule, space::AbstractFESpace;
                          dirichlet_tags=String[])
    el = mesh.element_type
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    i_diri = get_dirichlet_tags(mesh, dirichlet_tags)

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    global_dof = get_global_dof(mesh, space)
    local_matrix = zeros(T, n_dof_per_el, n_dof_per_el)
    for k in 1:n_el
        local_matrix .= zero(T)
        for q in eachindex(quad.weights)
            ∇φq = ∇φ(el, space, quad.points[q, :])
            for i in 1:n_dof_per_el, j in 1:n_dof_per_el
                local_matrix[i, j] += quad.weights[q] * dot(∇φq[i], ∇φq[j]) * jacs.dV[k]
            end
        end
        stamp!(global_matrix, local_matrix, global_dof[k, :], i_diri)
    end

    add_dirichlet!(global_matrix, i_diri)

    n_nodes = size(mesh.nodes, 1)
    return sparse_csc(global_matrix, n_nodes, n_nodes)
end