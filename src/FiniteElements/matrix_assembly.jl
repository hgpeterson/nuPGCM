function mass_matrix(mesh::Mesh, jacs::Jacobians, qr::QuadratureRule, sf::AbstractShapeFunction;
                     dirichlet_tags=String[])
    el = mesh.element_type

    i_diri = get_dirichlet_tags(mesh, dirichlet_tags)

    I = Int[]
    J = Int[]
    V = eltype(qr.weights)[]
    for k in axes(mesh.elements, 1)
        for q in eachindex(qr.weights)
            φq = φ(el, sf, qr.points[q, :])
            for i in axes(mesh.elements, 2)
                for j in axes(mesh.elements, 2)
                    if mesh.elements[k, i] in i_diri
                        continue
                    end
                    push!(I, mesh.elements[k, i])
                    push!(J, mesh.elements[k, j])
                    push!(V, qr.weights[q] * φq[i] * φq[j] * jacs.dV[k])
                end
            end
        end
    end

    for i in i_diri
        push!(I, i)
        push!(J, i)
        push!(V, 1)  # Dirichlet condition
    end

    return sparse(I, J, V, size(mesh.nodes, 1), size(mesh.nodes, 1))
end

function stiffness_matrix(mesh::Mesh, jacs::Jacobians, qr::QuadratureRule, sf::AbstractShapeFunction;
                          dirichlet_tags=String[])
    el = mesh.element_type

    i_diri = get_dirichlet_tags(mesh, dirichlet_tags)
    display(i_diri)

    I = Int[]
    J = Int[]
    V = eltype(qr.weights)[]
    for k in axes(mesh.elements, 1)
        for q in eachindex(qr.weights)
            ∇φq = ∇φ(el, sf, qr.points[q, :])
            for i in axes(mesh.elements, 2)
                if mesh.elements[k, i] in i_diri
                    continue
                end
                for j in axes(mesh.elements, 2)
                    push!(I, mesh.elements[k, i])
                    push!(J, mesh.elements[k, j])
                    push!(V, qr.weights[q] * dot(∇φq[i], ∇φq[j]) * jacs.dV[k])
                end
            end
        end
    end

    for i in i_diri
        push!(I, i)
        push!(J, i)
        push!(V, 1)  # Dirichlet condition
    end

    return sparse(I, J, V, size(mesh.nodes, 1), size(mesh.nodes, 1))
end