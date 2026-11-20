struct Mesh{G}
    grid::G
    surface_tag::String
    periodic_facets::Vector{PeriodicFacetPair}
end

function Base.summary(mesh::Mesh)
    t = typeof(mesh)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, mesh::Mesh)
    println(io, summary(mesh), ":")
    print(io, "├── grid: ")
    show(io, MIME"text/plain"(), mesh.grid)
    println(io)
    println(io, "├── surface_tag: \"", mesh.surface_tag, "\"")
      print(io, "└── periodic facet pairs: ", length(mesh.periodic_facets))
end

"""
    mesh = Mesh(ifile; surface_tag="surface")

Load a Gmsh `.msh` file and return a `Mesh`. Every boundary entity that Gmsh meshed
as the periodic copy of a master entity is paired with it facet by facet: the master
facets are the Ferrite mirror (constrained) facets and the copy facets the images.
"""
function Mesh(ifile; surface_tag="surface")
    @info "Loading mesh from $ifile..."
    should_finalize = FerriteGmsh.Gmsh.initialize()
    @time begin
    gmsh.open(ifile)
    grid, lookup = read_grid()
    periodic_facets = read_periodic_facets(grid, lookup)
    end
    should_finalize && FerriteGmsh.Gmsh.finalize()
    return Mesh(grid, surface_tag, periodic_facets)
end

"""
    grid, lookup = read_grid()

The `Ferrite.Grid` of the model open in Gmsh, with a facet set for each physical group of
facets, and `lookup`, the `FacetIndex` of each grid facet by its sorted vertices.
"""
function read_grid()
    dim = Int(gmsh.model.getDimension())
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    nodes = FerriteGmsh.tonodes()
    cells, gmsh_cells = FerriteGmsh.toelements(dim)
    FerriteGmsh.reorient!(cells, nodes)

    lookup = Dict{NTuple{dim, Int}, FacetIndex}()
    for (ci, cell) in enumerate(cells), (fi, fnodes) in enumerate(Ferrite.facets(cell))
        lookup[facet_key(Val(dim), fnodes)] = FacetIndex(ci, fi)
    end
    facetsets = Dict{String, Set{FacetIndex}}()
    for (_, group) in gmsh.model.getPhysicalGroups(dim - 1)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim - 1, group)
        facetsets[gmsh.model.getPhysicalName(dim - 1, group)] =
            mapreduce(entity -> entity_facets(dim - 1, entity, lookup), union, entities; init=Set{FacetIndex}())
    end
    grid = Grid(cells, nodes; facetsets, cellsets=FerriteGmsh.tocellsets(dim, gmsh_cells))
    return grid, lookup
end

"""
    pairs = read_periodic_facets(grid, lookup)

Periodic facet pairs of `grid`, built by `read_grid()` from the model open in Gmsh.
"""
function read_periodic_facets(grid, lookup)
    dim  = Int(gmsh.model.getDimension())
    tags = [tag for (_, tag) in gmsh.model.getEntities(dim - 1)]
    masters = gmsh.model.mesh.getPeriodic(dim - 1, tags)
    pairs = PeriodicFacetPair[]
    any(masters .!= tags) || return pairs

    for (tag, master) in zip(tags, masters)
        master == tag && continue
        # Gmsh stores x_copy = R x_master + t (row-major 4×4); Ferrite wants the map copy → master
        affine = reshape(gmsh.model.mesh.getPeriodicNodes(dim - 1, tag)[4], 4, 4)'
        R⁻¹ = inv(Tensor{2, dim}((i, j) -> affine[i, j]))
        t   = Vec{dim}(i -> affine[i, 4])
        mirror = entity_facets(dim - 1, master, lookup)
        image  = entity_facets(dim - 1, tag, lookup)
        collect_periodic_facets!(pairs, grid, mirror, image, x -> R⁻¹ ⋅ (x - t))
        @info "Periodic boundary: Gmsh entity $tag is a copy of entity $master"
    end
    return pairs
end

function facet_key(::Val{dim}, fnodes) where {dim}
    vertices = sort!([Int(fnodes[i]) for i in 1:dim])
    return ntuple(i -> vertices[i], Val(dim))
end

"""
    facets = entity_facets(fdim, tag, lookup)

Grid facets on the Gmsh entity `(fdim, tag)`.
"""
function entity_facets(fdim, tag, lookup::Dict{NTuple{dim, Int}, FacetIndex}) where {dim}
    facets = Set{FacetIndex}()
    types, _, nodetags = gmsh.model.mesh.getElements(fdim, tag)
    for (type, element_nodes) in zip(types, nodetags)
        nnodes = gmsh.model.mesh.getElementProperties(type)[4]
        for k in 1:nnodes:length(element_nodes)
            push!(facets, lookup[facet_key(Val(dim), @view element_nodes[k:k+nnodes-1])])
        end
    end
    return facets
end

### utility functions

"""
    p, t = get_p_t(grid)
    p, t = get_p_t(mesh)
    p, t = get_p_t(fname)

Return node coordinate matrix `p` (nnodes × sdim) and connectivity matrix `t`
(ncells × nodes_per_cell).
"""
function get_p_t(grid::Ferrite.AbstractGrid)
    nnodes = getnnodes(grid)
    sdim = length(grid.nodes[1].x)
    p = [grid.nodes[i].x[j] for i in 1:nnodes, j in 1:sdim]
    ncells = getncells(grid)
    nn = length(grid.cells[1].nodes)
    t = [grid.cells[k].nodes[j] for k in 1:ncells, j in 1:nn]
    return p, t
end
get_p_t(mesh::Mesh) = get_p_t(mesh.grid)
get_p_t(fname::AbstractString) = get_p_t(FerriteGmsh.togrid(fname))

"""
    p_to_t = get_p_to_t(t, np)

Returns a vector-of-vectors `p_to_t` where `p_to_t[i]` lists all `[k, j]` index
pairs in `t` that point to the `i`-th node.
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

Find all unique edges in the triangulation `t` (ne × 2 array).
`boundary_indices` indexes boundary edges; `emap` (nt × 3) maps local triangle
edges to global edge indices.
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
    h = median_edge_length(mesh)

Median length of the mesh edges.
"""
function median_edge_length(mesh::Mesh)
    p, t = get_p_t(mesh)
    edges, _, _ = all_edges(t)
    hs = sort([norm(p[edges[i, 1], :] - p[edges[i, 2], :]) for i in axes(edges, 1)])
    return hs[length(hs) ÷ 2]
end

"""
    h_cells = compute_h_cells(mesh)

Return the maximum edge length per cell.
"""
function compute_h_cells(mesh::Mesh)
    grid = mesh.grid
    return [maximum(norm(grid.nodes[cell.nodes[i]].x - grid.nodes[cell.nodes[j]].x)
                    for i in 1:length(cell.nodes)
                    for j in i+1:length(cell.nodes))
            for cell in grid.cells]
end
