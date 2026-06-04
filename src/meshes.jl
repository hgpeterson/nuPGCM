struct Mesh{G}
    grid::G
    surface_tag::String
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
      print(io, "└── surface_tag: \"", mesh.surface_tag, "\"")
end

"""
    mesh = Mesh(ifile; surface_tag="surface")

Load a Gmsh `.msh` file and return a `Mesh`.
"""
function Mesh(ifile; surface_tag="surface")
    @info "Loading mesh from $ifile..."
    @time grid = FerriteGmsh.togrid(ifile)
    return Mesh(grid, surface_tag)
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
