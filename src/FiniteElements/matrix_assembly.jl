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

function stamp!(global_matrix::SparseIJV, local_matrix, element_dof, i_diri)
    for i_local in axes(local_matrix, 1)
        i_global = element_dof[i_local]
        if i_global in i_diri
            continue
        end
        for j_local in axes(local_matrix, 2)
            j_global = element_dof[j_local]
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

function get_i_diri(dof_data::DoFData, dirichlet)
    T = eltype(dof_data.global_dof)
    i_diri = T[]
    for boundary in dirichlet
        i_diri = vcat(i_diri, dof_data.boundary_dofs[boundary])
    end
    return unique(i_diri)
end

function mass_matrix(mesh::Mesh, 
                     jacs::Jacobians, 
                     quad::QuadratureRule, 
                     space::AbstractFESpace,
                     dof_data::DoFData;
                     dirichlet=String[])
    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    i_diri = get_i_diri(dof_data, dirichlet)

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    global_dof = dof_data.global_dof
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

function rhs_vector(fe_data::FEData,
                    space::AbstractFESpace, 
                    dof_data::DoFData,
                    f, 
                    g;
                    dirichlet=String[])
    # unpack
    mesh = fe_data.mesh
    jacs = fe_data.jacobians
    quad = fe_data.quad_rule

    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    T = eltype(quad.weights)
    global_dof = dof_data.global_dof
    N = dof_data.N
    rhs = zeros(T, N)
    local_vector = zeros(T, n_dof_per_el)
    for k in 1:n_el
        local_vector .= zero(T)
        for q in eachindex(quad.weights)
            φq = φ(el, space, quad.points[q, :])
            x = transform_from_reference(el, jacs.∂x∂ξ[k, :, :], quad.points[q, :], mesh.nodes[mesh.elements[k, :], :])
            fq = f(x)
            for i in 1:n_dof_per_el
                local_vector[i] += quad.weights[q] * φq[i] * fq * jacs.dV[k]
            end
        end
        rhs[global_dof[k, :]] .+= local_vector
    end

    i_diri = get_i_diri(dof_data, dirichlet)
    for i in i_diri
        if i ≤ size(mesh.nodes, 1)
            # node position
             x = mesh.nodes[i, :]
        else
            # midpoint position
            edge_index = i - size(mesh.nodes, 1)
            n1, n2 = mesh.edges[edge_index, :]
            x = (mesh.nodes[n1, :] + mesh.nodes[n2, :])/2
        end
        rhs[i] = g(x)
    end

    return rhs
end

function stiffness_matrix(fe_data::FEData, 
                          space::AbstractFESpace,
                          dof_data::DoFData;
                          dirichlet=String[])
    # unpack
    mesh = fe_data.mesh
    jacs = fe_data.jacobians
    quad = fe_data.quad_rule

    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = n_dofs(el, space)

    i_diri = get_i_diri(dof_data, dirichlet)

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    global_dof = dof_data.global_dof
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

function elasticity_matrix(fe_data::FEData,
                           space::AbstractFESpace,
                           dof_data::DoFData;
                           dirichlet=String[],
                           dirichlet_mask=Tuple[],  # TODO: make it so you can pick which components have dirichlet conditions
                           d=3)
    # unpack
    mesh = fe_data.mesh
    jacs = fe_data.jacobians
    quad = fe_data.quad_rule

    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_dof_per_el = d*n_dofs(el, space)

    i_diri0 = get_i_diri(dof_data, dirichlet)

    # for velocity vector
    base(i) = mod1(i, n_dofs(el, space))
    comp(i) = 1 + div(i-1, n_dofs(el, space))
    n = maximum(dof_data.global_dof)
    global_dof(k, i) = dof_data.global_dof[k, base(i)] + n*(comp(i) - 1)
    i_diri = copy(i_diri0)
    for i in 1:d-1
        i_diri = [i_diri; i*n .+ i_diri0]
    end

    T = eltype(quad.weights)
    global_matrix = SparseIJV(T)
    local_matrix = zeros(T, n_dof_per_el, n_dof_per_el)
    for k in 1:n_el
        local_matrix .= zero(T)
        for q in eachindex(quad.weights)
            ∇φq = ∇φ(el, space, quad.points[q, :])
            ∇φq .= [jacs.∂ξ∂x[k, :, :]'*∇φq[i] for i in eachindex(∇φq)]  # need to transform derivatives
            for i in 1:n_dof_per_el
                comp_i = comp(i)
                base_i = base(i)
                for j in 1:n_dof_per_el
                    comp_j = comp(j)
                    base_j = base(j)
                    # ∂ᵢvᵢ∂ⱼuⱼ
                    local_matrix[i, j] += quad.weights[q] * ∇φq[base_i][comp_i]*∇φq[base_j][comp_j] * jacs.dV[k]
                    # ∂ⱼvᵢ∂ᵢuⱼ
                    local_matrix[i, j] += quad.weights[q] * ∇φq[base_i][comp_j]*∇φq[base_j][comp_i] * jacs.dV[k]
                    # ∂ₖvᵢ∂ₖuⱼ
                    if comp_i == comp_j 
                        local_matrix[i, j] += quad.weights[q] * dot(∇φq[base_i], ∇φq[base_j]) * jacs.dV[k]
                    end
                end
            end
        end
        stamp!(global_matrix, local_matrix, global_dof.(k, 1:n_dof_per_el), i_diri)
    end

    add_dirichlet!(global_matrix, i_diri)

    return sparse_csc(global_matrix, d*n, d*n)
end