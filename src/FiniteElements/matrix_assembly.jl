struct SparseIJV{VI, VF}
    I::VI  # row indices
    J::VI  # column indices
    V::VF  # values
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

################################################################################

function mass_matrix(mesh::Mesh, jacs::Jacobians, quad::QuadratureRule, space::AbstractFESpace;
                     dirichlet=String[])
    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    i_diri = get_dirichlet_tags(mesh, dirichlet)

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

    n = maximum(global_dof)
    return sparse_csc(global_matrix, n, n)
end

function rhs_vector(mesh::Mesh, jacs::Jacobians, quad::QuadratureRule, space::AbstractFESpace, f, g;
                    dirichlet=String[])
    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    T = eltype(quad.weights)
    global_dof = get_global_dof(mesh, space)
    n = maximum(global_dof) 
    rhs = zeros(T, n)
    local_vector = zeros(T, n_dof_per_el)
    for k in 1:n_el
        local_vector .= zero(T)
        for q in eachindex(quad.weights)
            φq = φ(el, space, quad.points[q, :])
            x = transform_from_reference(el, jacs.∂x∂ξ[k], quad.points[q, :], mesh.nodes[mesh.elements[k, :], :])
            fq = f(x)
            for i in 1:n_dof_per_el
                local_vector[i] += quad.weights[q] * φq[i] * fq * jacs.dV[k]
            end
        end
        rhs[global_dof[k, :]] .+= local_vector
    end

    i_diri = get_dirichlet_tags(mesh, dirichlet)
    for i in i_diri
        rhs[i] = g(mesh.nodes[i, :])
    end

    return rhs
end

function stiffness_matrix(mesh::Mesh, jacs::Jacobians, quad::QuadratureRule, space::AbstractFESpace;
                          dirichlet=String[])
    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    i_diri = get_dirichlet_tags(mesh, dirichlet)

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    global_dof = get_global_dof(mesh, space)
    local_matrix = zeros(T, n_dof_per_el, n_dof_per_el)
    for k in 1:n_el
        local_matrix .= zero(T)
        for q in eachindex(quad.weights)
            ∇φq = ∇φ(el, space, quad.points[q, :])
            ∇φq = [jacs.∂ξ∂x[k, :, :]'*∇φq[i] for i in eachindex(∇φq)]  # need to transform derivatives
            for i in 1:n_dof_per_el, j in 1:n_dof_per_el
                local_matrix[i, j] += quad.weights[q] * dot(∇φq[i], ∇φq[j]) * jacs.dV[k]
            end
        end
        stamp!(global_matrix, local_matrix, global_dof[k, :], i_diri)
    end

    add_dirichlet!(global_matrix, i_diri)

    n = maximum(global_dof)
    return sparse_csc(global_matrix, n, n)
end

function stokes_matrix(mesh::Mesh, jacs::Jacobians, quad::QuadratureRule, 
                       u_space::AbstractFESpace,
                       v_space::AbstractFESpace,
                       w_space::AbstractFESpace,
                       p_space::AbstractFESpace; 
                       u_dirichlet=String[],
                       v_dirichlet=String[],
                       w_dirichlet=String[])
    el = mesh.element_type
    n_el = size(mesh.elements, 1)
    n_dof_per_el_u = n_dofs(el, u_space)
    n_dof_per_el_v = n_dofs(el, v_space)
    n_dof_per_el_w = n_dofs(el, w_space)
    n_dof_per_el_p = n_dofs(el, p_space)

    i_diri_u = get_dirichlet_tags(mesh, u_dirichlet)
    i_diri_v = get_dirichlet_tags(mesh, v_dirichlet)
    i_diri_w = get_dirichlet_tags(mesh, w_dirichlet)

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    global_dofs_u = get_global_dof(mesh, u_space)
    global_dofs_v = get_global_dof(mesh, v_space) .+ maximum(global_dofs_u)
    global_dofs_w = get_global_dof(mesh, w_space) .+ maximum(global_dofs_v)
    global_dofs_p = get_global_dof(mesh, p_space) .+ maximum(global_dofs_w)
    local_matrix = zeros(T, n_dof_per_el_v + n_dof_per_el_p, n_dof_per_el_v + n_dof_per_el_p) #TODO: ???
    for k in 1:n_el
        local_matrix .= zero(T)
        for q in eachindex(quad.weights)
            φq_u = φ(el, u_space, quad.points[q, :])
            φq_v = φ(el, v_space, quad.points[q, :])
            φq_w = φ(el, w_space, quad.points[q, :])
            φq_p = φ(el, p_space, quad.points[q, :])
            ∇φq_u = ∇φ(el, u_space, quad.points[q, :])
            ∇φq_v = ∇φ(el, v_space, quad.points[q, :])
            ∇φq_w = ∇φ(el, w_space, quad.points[q, :])
            ∇φq_p = ∇φ(el, p_space, quad.points[q, :])
            for i in 1:n_dof_per_el_u, j in 1:n_dof_per_el_p
                # v⋅∇p 
                local_matrix[i, j] += quad.weights[q] * (φq_u[i]*∇φq_p[1][j] + 
                                                         φq_v[i]*∇φq_p[2][j] + 
                                                         φq_w[i]*∇φq_p[3][j]) * jacs.dV[k]
            end
            for i in 1:n_dof_per_el_p, j in 1:n_dof_per_el_u
                # q∇⋅u
                local_matrix[i, j] += quad.weights[q] * (φq_p[i]*(∇φq_u[1][j] +
                                                                  ∇φq_v[2][j] +
                                                                  ∇φq_w[3][j])) * jacs.dV[k]
            end
            for i in 1:n_dof_per_el_u, j in 1:n_dof_per_el_u
                # ∇v⊙∇u (constant viscosity for now)
                local_matrix[i, j] += quad.weights[q] * (dot(∇φq_u[i], ∇φq_u[j]) + 
                                                         dot(∇φq_v[i], ∇φq_v[j]) + 
                                                         dot(∇φq_w[i], ∇φq_w[j])) * jacs.dV[k]
            end
        end
        stamp!(global_matrix, local_matrix,
               vcat(global_dof_v[k, :], global_dof_p[k, :]), i_diri)
    end

    add_dirichlet!(global_matrix, i_diri)

    n = maximum(vcat(global_dof_v, global_dof_p))
    return sparse_csc(global_matrix, n, n)
end