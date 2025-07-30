struct Mesh{M, S, D, O, DO, G, DG}
    model::M         # unstructured discrete model
    spaces::S        # trial and test spaces for velocity, pressure, and buoyancy
    dofs::D          # degree of freedom handler
    Ω::O             # triangulation
    dΩ::DO           # measure
    Γ::G             # surface boundary triangulation
    dΓ::DG           # surface boundary measure
    dim::Int         # dimension of the problem
end

"""
    m = Mesh(ifile, b₀; degree=4)

Build a struct holding mesh-related data.

`degree` is the degree of integration for the measures `dΩ` and `dΓ`.
`b₀` is a function for the buoyancy surface boundary condition.
"""
function Mesh(ifile, b₀; degree=4)
    model = GmshDiscreteModel(ifile)
    spaces = Spaces(model, b₀)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(model, tags=["sfc"])
    dΓ = Measure(Γ, degree)
    dofs = DoFHandler(spaces, dΩ)
    if model.grid_topology.polytopes[1] == TRI
        dim = 2
    elseif model.grid_topology.polytopes[1] == TET
        dim = 3
    else
        throw(ArgumentError("Could not determine dimension of mesh."))
    end
    return Mesh(model, spaces, dofs, Ω, dΩ, Γ, dΓ, dim)
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

@doc raw"""
    circumdiameter(vertices)

Compute diameter of circumscribed circle (for triangle) or sphere (for tetrahedron).

We use the formulae from 
[this math stackexchange answer](https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron).
For a triangle of side lengths `a`, `b`, and `c`, the circumdiameter is given by
```math
D = abc / (2A)
```
where `A` is the area of the triangle.
For a tetrahedron with edge lengths `a`, `b`, `c`, `a_1`, `b_1`, and `c_1`, the circumdiameter is given by
```math
D = \sqrt{p(p - aa_1)(p - bb_1)(p - cc_1)}/(3V) \quad \text{where} \quad p = \frac{aa_1 + bb_1 + cc_1}{2}
```
where `V` is the volume of the tetrahedron.

The `vertices` argument is assumed to be a matrix of size `n x 3` where `n` is the number of vertices.
"""
function circumdiameter(vertices)
    if size(vertices, 2) != 3
        throw(ArgumentError("Expected vertices to have 3 columns, got $(size(vertices, 2))."))
    end

    # number of vertices
    n = size(vertices, 1)

    if n == 3  # triangle
        a = norm(vertices[1, :] - vertices[2, :])
        b = norm(vertices[2, :] - vertices[3, :])
        c = norm(vertices[3, :] - vertices[1, :])
        s = (a + b + c) / 2
        A = sqrt(s * (s - a) * (s - b) * (s - c)) # Heron's formula
        return a * b * c / (2 * A)
    elseif n == 4  # tetrahedron
        a = norm(vertices[1, :] - vertices[2, :])
        b = norm(vertices[1, :] - vertices[3, :])
        c = norm(vertices[1, :] - vertices[4, :])
        a1 = norm(vertices[3, :] - vertices[4, :])
        b1 = norm(vertices[2, :] - vertices[4, :])
        c1 = norm(vertices[2, :] - vertices[3, :])
        p = (a * a1 + b * b1 + c * c1) / 2
        V = abs(det(hcat(vertices[2, :] - vertices[1, :], vertices[3, :] - vertices[1, :], vertices[4, :] - vertices[1, :]))) / 6
        return sqrt(p * (p - a * a1) * (p - b * b1) * (p - c * c1)) / (3 * V)
        # # This method is slightly faster but not as clean
        # p1 = vertices[1, :]
        # p2 = vertices[2, :]
        # p3 = vertices[3, :]
        # p4 = vertices[4, :]
        # A = 2 * [(p2[1] - p1[1]) (p2[2] - p1[2]) (p2[3] - p1[3])
        #          (p3[1] - p1[1]) (p3[2] - p1[2]) (p3[3] - p1[3])
        #          (p4[1] - p1[1]) (p4[2] - p1[2]) (p4[3] - p1[3])]
        # b = [p2[1]^2 - p1[1]^2 + p2[2]^2 - p1[2]^2 + p2[3]^2 - p1[3]^2
        #         p3[1]^2 - p1[1]^2 + p3[2]^2 - p1[2]^2 + p3[3]^2 - p1[3]^2;
        #         p4[1]^2 - p1[1]^2 + p4[2]^2 - p1[2]^2 + p4[3]^2 - p1[3]^2]
        # center = A \ b
        # return 2 * norm(center - p1) # circumdiameter is twice the circumradius
    else
        throw(ArgumentError("Expected 3 or 4 vertices, got $n."))
    end
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