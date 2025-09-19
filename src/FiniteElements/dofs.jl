struct DoFData{M, D}
    global_dof::M    # global_dof[k, i] = global dof of the i-th local dof of element k
    boundary_dofs::D  # boundary_dofs[key] = vector of global dofs on boundary with given key
end

function DoFData(mesh::Mesh, ::P1)
    global_dof = mesh.elements
    boundary_dofs = mesh.boundary_nodes
    return DoFData(global_dof, boundary_dofs)
end

function DoFData(mesh::Mesh, ::Mini)
    n_nodes = size(mesh.nodes, 1)
    n_el = size(mesh.elements, 1)
    global_dof = hcat(mesh.elements, n_nodes .+ collect(1:n_el))
    boundary_dofs = mesh.boundary_nodes
    return DoFData(global_dof, boundary_dofs)
end

function DoFData(mesh::Mesh, space::P2)
    el = get_element_type(mesh)
    n_el = size(mesh.elements, 1)
    n_nodes_per_el = size(mesh.elements, 2)
    n_dofs_per_el = n_dofs(el, space)
    n_nodes = size(mesh.nodes, 1)

    T = eltype(mesh.elements)
    global_dof = zeros(T, n_el, n_dofs_per_el)
    global_dof[:, 1:n_nodes_per_el] .= mesh.elements
    for k in 1:n_el
        global_dof[k, n_nodes_per_el+1:end] = n_nodes .+ mesh.emap[k, :]
    end

    boundary_dofs = Dict{String, Vector{T}}()
    for boundary in keys(mesh.boundary_nodes)
        boundary_dofs[boundary] = vcat(mesh.boundary_nodes[boundary],
                                       n_nodes .+ mesh.boundary_edges[boundary])
    end

    return DoFData(global_dof, boundary_dofs)
end