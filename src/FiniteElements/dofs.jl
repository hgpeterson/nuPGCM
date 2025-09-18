# global_dof[k, i] gives the global dof number of the i-th local dof of element k
function get_global_dof(mesh::Mesh, ::P1)
    return mesh.elements
end
function get_global_dof(mesh::Mesh, ::Bubble)
    n_elements = size(mesh.elements, 1)
    return reshape(collect(1:n_elements), n_elements, 1)
end
function get_global_dof(mesh::Mesh, ::Mini)
    n_nodes = size(mesh.nodes, 1)
    n_elements = size(mesh.elements, 1)
    return hcat(mesh.elements, n_nodes .+ collect(1:n_elements))
end
function get_global_dof(mesh::Mesh, space::P2)
    n_el = size(mesh.elements, 1)
    n_nodes_per_el = size(mesh.elements, 2)
    element_type = get_element_type(mesh)
    n_dof_per_el = n_dofs(element_type, space)

    global_dof = zeros(eltype(mesh.elements), n_el, n_dof_per_el)
    global_dof[:, 1:n_nodes_per_el] .= mesh.elements
    for k in 1:n_el
        global_dof[k, n_nodes_per_el+1:end] = size(mesh.nodes, 1) .+ mesh.emap[k, :]
    end
    return global_dof
end

# global_dof[j][k, i] gives the global dof number of the i-th local dof of element k for space j
function get_global_dof(mesh::Mesh, spaces::AbstractVector{<:AbstractFESpace})
    return [get_global_dof(mesh, space) for space in spaces]
end
