struct Mesh{M, O, DO}
    model::M         # unstructured discrete model
    spaces::Spaces   # trial and test spaces for velocity, pressure, and buoyancy
    dofs::DoFHandler # degree of freedom handler
    Ω::O             # triangulation
    dΩ::DO           # measure
    dim::Int         # dimension of the problem
end

"""
    m = Mesh(ifile)

Returns a struct holding mesh-related data.
"""
function Mesh(ifile)
    model = GmshDiscreteModel(ifile)
    spaces = Spaces(model)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)
    dofs = DoFHandler(spaces, dΩ)
    if model.grid_topology.polytopes[1] == TRI
        dim = 2
    elseif model.grid_topology.polytopes[1] == TET
        dim = 3
    else
        throw(ArgumentError("Could not determine dimension of mesh."))
    end
    return Mesh(model, spaces, dofs, Ω, dΩ, dim)
end

### some utility functions for working with meshes

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