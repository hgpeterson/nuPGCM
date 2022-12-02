struct FEGrid{FM<:AbstractMatrix, IM<:AbstractMatrix, IV<:AbstractVector, IN<:Integer}
    # order of shape functions on this grid
    order::IN

    # dimension of space
    dim::IN

    # shape functions on this grid
    s::ShapeFunctions

    # node positions
    p::FM # float matrix

    # number of nodes
    np::IN

    # node indices defining each element
    t::IM # integer matrix

    # number of elements
    nt::IN

    # number of nodes per element
    nn::IN

    # edge node indices
    e::IV # integer vector

    # number of edge nodes
    ne::IN
end

"""
    g = FEGrid(p, t, e, order)

Construct a FE grid of order `order` with points `p`, elements `t`, and boundary nodes `e`.
"""
function FEGrid(p, t, e, order::IN) where IN <: Integer
    # dimension of space
    dim = size(p, 2)

    # order of grid
    if order == 0
        # only need centroids
        pp = zeros(size(t, 1), dim)
        ee = Vector{IN}()
        for k in axes(t, 1)
            pp[k, :] = 1/(dim + 1)*sum(p[t[k, :], :], dims=1)
            if sum([t[k, i] in e for i in axes(t, 2)]) > 0 
                # at least one of the points in this element is on the boundary, make centroid a bounday node
                push!(ee, k)
            end
        end
        p = pp
        t = zeros(IN, size(t, 1), 1)
        t[:, 1] = 1:size(t, 1)
        e = ee
    elseif order == 1
        # grid already order 1
    elseif order > 1
        # add nodes for higher orders
        p, t, e = add_nodes(p, t, e, order)
    else
        error("Unsupported grid order `$order`.")
    end

    # indices
    np = size(p, 1)
    nt = size(t, 1)
    nn = size(t, 2)
    ne = size(e, 1)

    # setup shape functions
    s = ShapeFunctions(order, dim)

    return FEGrid(order, dim, s, p, np, t, nt, nn, e, ne)
end
function FEGrid(gfile, order)
    # read grid data
    p, t, e = read_gfile_h5(gfile)

    return FEGrid(p, t, e, order) 
end

"""
    p, t, e = read_gfile_h5(gfile)

Load grid from HDF5 file `gfile`.
"""
function read_gfile_h5(gfile)
    file = h5open(gfile, "r")
    p = read(file, "p")
    t = read(file, "t")
    e = read(file, "e")
    close(file)
    t = convert(Matrix{Int64}, t)
    try
        e = convert(Vector{Int64}, e)
    catch
        # for some old meshes...
        e = e[:, 1]
        e = convert(Vector{Int64}, e)
    end
    return p, t, e
end

"""
	p, t, e = add_nodes(p, t, e, order)

Add nodes to mesh for higher-order shape functions.
"""
function add_nodes(p, t, e, order)
    # get edges
    edges, boundary_indices, emap = all_edges(t, e)

    # dimension of space
    dim = size(p, 2)

    if order == 2
        # add midpoints
        np0 = size(p, 1)
        new_pts = 1/2*reshape(p[edges[:, 1], :] + p[edges[:, 2], :], (size(edges, 1), dim))
        pnew = [p; new_pts]

        # map to triangle data structure
        tnew = hcat(t, np0 .+ emap)

        # add points that were on the boundary to `e`
        enew = [e; np0 .+ boundary_indices]
    elseif order == 3 && dim == 2
        # number of nodes per triangle
        n = 10

        # first add 1/3 points
        np0 = size(p, 1)
        new_pts = reshape(p[edges[:, 1], :] + 1/3*(p[edges[:, 2], :] - p[edges[:, 1], :]), (size(edges, 1), 2))
        pnew = [p; new_pts]
        np1 = size(pnew, 1)
        # then add 2/3 points
        new_pts = reshape(p[edges[:, 1], :] + 2/3*(p[edges[:, 2], :] - p[edges[:, 1], :]), (size(edges, 1), 2))
        pnew = [pnew; new_pts]
        np2 = size(pnew, 1)
        # finally add center points
        new_pts = reshape(1/3*(p[t[:, 1], :] + p[t[:, 2], :] + p[t[:, 3], :]), (size(t, 1), 2))
        pnew = [pnew; new_pts]

        # not as easy to determine the indices for each triangle because the 1/3rd point for one triangle is
        # the 2/3rd point for another... this works but it is slow
        tnew = zeros(Int64, size(t, 1), n)
        ps = reference_element_nodes(order, 2)
        tnew[:, 1:3] = t
        @showprogress "Triangulating 3rd-order mesh..." for k in axes(t, 1)
            for i=4:n-1
                p₀ = transform_from_ref_el(ps[i, :], pnew[t[k, :], :])
                idx = get_idx(pnew, p₀)
                tnew[k, i] = idx
            end
        end
        tnew[:, 10] = np2 .+ (1:size(t,1))'

        # add points that were on boundary to `e`
        enew = [e; np0 .+ boundary_indices]
        enew = [enew; np1 .+ boundary_indices]
    else
        error("Unsupported grid order `$order` for dimension `$dim`.")
    end

    return pnew, tnew, enew
end

"""
    i = get_idx(p, p₀)

Find the node index of point `p₀` in set of points `p`.
"""
function get_idx(p, p₀)
    Δp = @. (p[:, 1] - p₀[1])^2 + (p[:, 2] - p₀[2])^2
    return argmin(Δp)
end

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique `edges` (ne x 2 array) in the triangulation `t`.
Second output is indices to the boundary edges.
Third output `emap` (nt x dim+1 array) is a mapping from local edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in elemnt it.
"""
function all_edges(t, e)
    # dimension of space
    dim = size(t, 2) - 1

    # get all possible edge index pairs
    ne = Int64((dim + 1)*dim/2) # number of edges per element = dim + 1 choose 2
    pairs = [1 2
             2 3
             1 3
             1 4
             2 4
             3 4]

    # find all edges
    etag = t[:, pairs[1, :]]
    for i=2:ne
        etag = vcat(etag, t[:, pairs[i, :]])
    end

    # order node indices so lowest ones are in first column, tag each edge with its global index in 3rd column
    etag = hcat(sort(etag, dims=2), 1:ne*size(t, 1))

    # now sort so that first column goes from lowest to highest node index
    etag = sortslices(etag, dims=1)

    # determine if edge is a duplicate or should be kept
    keep = unique(i -> etag[i, 1:2], 1:size(etag, 1))

    # only keep unique edges
    edges = etag[keep, 1:2]

    # boundary edges have both nodes in `e`
    bndix = Vector{Int64}([])
    for i in axes(edges, 1)
        if edges[i, 1] in e && edges[i, 2] in e
            push!(bndix, i)
        end
    end

    # compute mapping to global indices
    kept = zeros(Bool, ne*size(t, 1))
    kept[keep] .= 1
    emap = cumsum(kept)
    invpermute!(emap, etag[:, 3])
    emap = reshape(emap, :, ne)

    return edges, bndix, emap
end

struct Jacobians{V<:AbstractVector, A<:AbstractArray}
    # |∂x/∂ξ| for each element
    dets::V

    # ∂ξ/∂x for each element
    Js::A
end

"""
    J = Jacobians(g)

Compute Jacobian terms for transformations from reference element to element on grid.
Given the vertices xᵢ ∈ ℜᵈ of the reference element, the transformation ξ ↦ x is 
    x = x₁ + A*ξ
where
    A in 1D = [x₂ - x₁],
    A in 2D = [x₂ - x₁  x₃ - x₁],
    A in 3D = [x₂ - x₁  x₃ - x₁  x₄ - x₁].
In other words, A = ∂x/∂ξ is the Jacobian. To transform from global coordinates to 
the reference element, we then need the inverse of A, J = ∂ξ/∂x:
    J in 1D = [∂ξ/∂x],
    J in 2D = [∂ξ/∂x  ∂ξ/∂y;  ∂η/∂x  ∂η/∂y],
    J in 3D = [∂ξ/∂x  ∂ξ/∂y  ∂ξ/∂z;  ∂η/∂x  ∂η/∂y  ∂η/∂z;  ∂ζ/∂x  ∂ζ/∂y  ∂ζ/∂z].
"""
function Jacobians(g::FEGrid)
    # pre-allocate
    dets = zeros(g.nt)
    Js = zeros(g.nt, g.dim, g.dim)

    # loop through elements in g
    for k=1:g.nt
        # build A
        A = zeros(g.dim, g.dim)
        for i=1:g.dim
            A[:, i] = g.p[g.t[k, i+1], :] - g.p[g.t[k, 1], :]
        end

        # compute determinant
        dets[k] = abs(det(A))

        # invert for J
        Js[k, :, :] = inv(A)
    end
    return Jacobians(dets, Js)
end
function Jacobians(gfile::String)
    # get order 1 FE grid
    g = FEGrid(gfile, 1)
    return Jacobians(g)
end