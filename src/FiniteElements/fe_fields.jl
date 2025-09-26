# struct to hold data for a FE field
struct FEField{F<:FEData, S<:AbstractFESpace, D<:DoFData, V<:AbstractArray}
    fe_data::F
    space::S
    dofs::D
    values::V
end

function FEField(fe_data::FEData, space::AbstractFESpace, dofs::DoFData)
    values = zeros(eltype(fe_data.mesh.nodes), dofs.N)
    return FEField(fe_data, space, dofs, values)
end

function L2_error(u::FEField, u₀)
    mesh = u.fe_data.mesh
    jacs = u.fe_data.jacobians
    quad = u.fe_data.quad_rule
    space = u.space
    dof_data = u.dofs

    # u at quadrature points
    el = get_element_type(mesh)
    n_dofs_per_el = n_dofs(el, space)
    uq = zeros(eltype(u.values), size(mesh.elements, 1), length(quad.weights))
    for k in 1:size(mesh.elements, 1)
        for q in eachindex(quad.weights)
            φq = φ(el, space, quad.points[q, :])
            for i in 1:n_dofs_per_el
                uq[k, q] += u.values[dof_data.global_dof[k, i]] * φq[i]
            end
        end
    end

    # integrate
    e = zero(eltype(u.values))
    for k in 1:size(mesh.elements, 1)
        for q in eachindex(quad.weights)
            x = transform_from_reference(el, jacs.∂x∂ξ[k, :, :], quad.points[q, :], mesh.nodes[mesh.elements[k, :], :])
            e += quad.weights[q] * (uq[k, q] - u₀(x))^2 * jacs.dV[k]
        end
    end

    return √e
end