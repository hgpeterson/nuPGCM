struct Mesh{P, T, PT}
    p::P
    t::T
    p_to_t::PT
end

"""
    m = Mesh(model::Gridap.Geometry.UnstructuredDiscreteModel)
    m = Mesh(model::Gridap.Geometry.UnstructuredDiscreteModel, bdy_name)
    m = Mesh(fname::AbstractString)
    m = Mesh(p, t)
    m = Mesh(p, t, p_to_t)

A simple custom struct to hold a mesh. `p` defines the node coordinates, 
`t` defines the connectivities, and `p_to_t` maps nodes to cells.
"""
function Mesh(model::Gridap.Geometry.UnstructuredDiscreteModel)
    p, t = get_p_t(model)
    return Mesh(p, t)
end
function Mesh(p, t)
    p_to_t = get_p_to_t(t, size(p, 1))
    return Mesh(p, t, p_to_t)
end
function Mesh(fname::AbstractString)
    model = GmshDiscreteModel(fname)
    return Mesh(model)
end
function Mesh(model::Gridap.Geometry.UnstructuredDiscreteModel, bdy_name)
    # determine entity tags on boundary called `bdy_name`
    tag = findfirst(model.face_labeling.tag_to_name .== bdy_name)
    entities = model.face_labeling.tag_to_entities[tag]

    # get indices of boundary triangles
    tri_tags = model.face_labeling.d_to_dface_to_entity[3]
    k_bdy = findall(k -> tri_tags[k] ∈ entities, 1:size(tri_tags, 1))

    # `t` data structure for boundary (note: these are still the global indices)
    tris = model.grid_topology.n_m_to_nface_to_mfaces[3, 1] # all triangles in the mesh
    t = [tris[k][i] for k ∈ k_bdy, i ∈ 1:3]

    # make `p` data structure for boundary (note: this is still _all_ of the nodes in the mesh)
    nc = model.grid.node_coordinates
    p = [nc[i][j] for i ∈ axes(nc, 1), j ∈ 1:length(nc[1])]

    return Mesh(p, t)
end

"""
    p, t = get_p_t(model::Gridap.Geometry.UnstructuredDiscreteModel)
    p, t = get_p_t(fname::AbstractString)

Return the node coordinates `p` and the connectivities `t` of a mesh.
"""
function get_p_t(model::Gridap.Geometry.UnstructuredDiscreteModel)
    # unpack node coords
    nc = model.grid.node_coordinates
    np = length(nc)
    d = length(nc[1])
    p = [nc[i][j] for i ∈ 1:np, j ∈ 1:d]

    # unpack connectivities
    cni = model.grid.cell_node_ids
    nt = length(cni)
    nn = length(cni[1])
    t = [cni[i][j] for i ∈ 1:nt, j ∈ 1:nn]

    return p, t
end
function get_p_t(fname::AbstractString)
    # load model
    model = GmshDiscreteModel(fname)
    return get_p_t(model)
end

"""
    p_to_t = get_p_to_t(t, np)

Returns a vector of vectors of vectors `p_to_t` such that p_to_t[i] lists
all the [k, j] pairs in `t` that point to the ith node of the mesh of size `np`.
"""
function get_p_to_t(t, np)
    p_to_t = [[] for i ∈ 1:np]
    for k ∈ axes(t, 1)
        for i ∈ axes(t, 2)
            push!(p_to_t[t[k, i]], [k, i])
        end
    end
    return p_to_t
end

"""
    u(x) = nan_eval(u, x)

Evaluate `u(x)` and return `NaN` if an error occurs.
"""
function nan_eval(u, x)
    try 
        u(x) 
    catch 
        NaN 
    end
end

"""
    u = unpack_fefunction(u, m::Mesh)

Unpack Gridap finite element function `u` into a vector of values at the nodes 
of the mesh. (Assumes `u` is continuous).
"""
function unpack_fefunction(u, m::Mesh)
    u_cell_values = get_cell_dof_values(u)
    return [u_cell_values[m.p_to_t[i][1][1]][m.p_to_t[i][1][2]] for i ∈ 1:size(m.p, 1)]

    # this works for order 1 spaces
    # return sortslices([U.space.metadata.free_dof_to_node       u.free_values
    #                    U.space.metadata.dirichlet_dof_to_node  U.dirichlet_values], dims=1)[:, 2]
end