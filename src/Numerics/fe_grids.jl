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
    edges, boundary_indices, emap = all_edges(t)

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
        enew = [e; np0 .+ boundary_indices; np1 .+ boundary_indices]
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

function all_faces(t)
    # form all faces
    ftag = [t[:, [1,2,3]]; t[:, [1,2,4]]; t[:, [1,3,4]]; t[:, [2,3,4]]]
    nfaces = size(ftag, 1)

    # sort columns and tag with global indices in 4th column
    ftag = [sort(ftag, dims=2)  1:nfaces]

    # sort rows
    ftag = sortslices(ftag, dims=1)

    # unique faces
    faces = unique(ftag[:, 1:3], dims=1)
    
    # indices of unique faces
    keep = zeros(Bool, nfaces)
    keep[unique(i -> ftag[i, 1:3], 1:nfaces)] .= 1

    # face `i` has no duplicates if `i` and `i+1` are in `keep`
    i_surf = findall(i -> keep[i]*keep[i+1], 1:nfaces-1)
    if keep[nfaces]
        i_surf = [i_surf; nfaces]
    end

    # non-duplicates are surface triangles
    surf_faces = ftag[i_surf, 1:3]

    # mapping from local to global face index
    fmap = cumsum(keep)
    invpermute!(fmap, ftag[:, 4])
    fmap = reshape(fmap, :, 4)
    return faces, surf_faces, fmap
end

"""
    emap, edges, e = all_edges(t)

Find all unique `edges` (ne x 2 array) in the triangulation `t`.
Second output is indices to the boundary edges.
Third output `emap` (nt x dim+1 array) is a mapping from local edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in element it.
"""
function all_edges(t)
    # find all edges
    etag = [t[:,[1,2]]; t[:,[2,3]]; t[:,[3,1]]]

    # sort columns and tag with global indices in 3rd column
    etag = [sort(etag, dims=2)  1:3*size(t,1)]

    # now sort so that first column goes from lowest to highest node index
    etag = sortslices(etag, dims=1)

    # determine if edge is a duplicate or should be kept
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]

    # only keep unique edges
    edges = etag[keep,1:2]

    # compute mapping to global indices
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)

    # boundary nodes don't share an edge
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    e = unique(edges[bndix, :][:])

    return emap, edges, e
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
    x = A*ξ + b
where
    A in 1D = (x₂ - x₁)/2,
    A in 2D = [x₂ - x₁  x₃ - x₁],
    A in 3D = [x₂ - x₁  x₃ - x₁  x₄ - x₁].
    b in 1D = (x₂ + x₁)/2,
    b in 2D = x₁
    b in 3D = x₁
In other words, A = ∂x/∂ξ is the Jacobian. To transform from global coordinates to 
the reference element, we then need the inverse of A, J = ∂ξ/∂x:
    J in 1D = ∂ξ/∂x,
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
        A = [g.p[g.t[k, j+1], i] - g.p[g.t[k, 1], i] for i=1:g.dim, j=1:g.dim]

        if g.dim == 1
            # A = (x₂ - x₁)/2 for 1D case
            A /= 2.0
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