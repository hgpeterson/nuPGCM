struct Mesh{M, O, DO, G, DG}
    model::M  # unstructured discrete model
    Ω::O      # triangulation
    dΩ::DO    # measure
    Γ::G      # surface boundary triangulation
    dΓ::DG    # surface boundary measure
end

function Base.summary(mesh::Mesh)
    t = typeof(mesh)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, mesh::Mesh)
    println(io, summary(mesh), ":")
    println(io, "├── model: ", mesh.model)
    println(io, "├── Ω: ", mesh.Ω)
    println(io, "├── dΩ: ", mesh.dΩ)
    println(io, "├── Γ: ", mesh.Γ)
      print(io, "└── dΓ: ", mesh.dΓ)
end

"""
    m = Mesh(ifile; degree=4, surface_tags=["surface"])

Build a struct holding mesh-related data.

`degree` is the degree of integration for the measures `dΩ` and `dΓ`.
"""
function Mesh(ifile; degree=4, surface_tags=["surface"])
    model = GmshDiscreteModel(ifile)
    @info "Building `Gridap.Triangulation`s and `Gridap.Measure`s..."
    @time begin
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(model, tags=surface_tags)
    dΓ = Measure(Γ, degree)
    end
    return Mesh(model, Ω, dΩ, Γ, dΓ)
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

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique edges in the triangulation `t` (ne x 2 array)
Second output is indices to the boundary edges.
Third output emap (nt x 3 array) is a mapping from local triangle edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in triangle it.
"""
function all_edges(t)
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
    e = boundary_nodes(t)

Find all boundary nodes in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

"""
    h_cells = compute_h_cells(mesh::Mesh)

Compute characteristic cell size cell in the mesh.

Here we use h_K ∼ vol(K)^(1/dim) as a simple measure of cell size. 
"""
function compute_h_cells(mesh::Mesh)
    dim = num_dims(mesh.model)
    vols = collect(get_cell_measure(mesh.Ω))
    return vols .^ (1 / dim)
end