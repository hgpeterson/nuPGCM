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
    # dimension of space
    dim = size(p, 2)

    emap, edges, bndix = all_edges(t)

    if dim == 1
        pnew = copy(p)
        tnew = copy(t)
        for i=2:order
            new_pts = reshape((order - i + 1)/order*p[edges[:, 1], :] + (i - 1)/order*p[edges[:, 2], :], (size(edges, 1), :))
            pnew = [pnew; new_pts]
            tnew = hcat(tnew, size(p, 1) + (i - 2)*(size(p, 1) - 1) .+ emap)
        end
        enew = e
    else
        if order == 2
            # add midpoints
            np0 = size(p, 1)
            new_pts = 1/2*reshape(p[edges[:, 1], :] + p[edges[:, 2], :], (size(edges, 1), :))
            pnew = [p; new_pts]

            # map to triangle data structure
            tnew = hcat(t, np0 .+ emap)

            # add points that were on the boundary to `e`
            enew = [e; np0 .+ bndix]
        elseif order == 3 && dim == 2
                # number of nodes per triangle
                n = 10

                # first add 1/3 points
                np0 = size(p, 1)
                new_pts = reshape(2/3*p[edges[:, 1], :] + 1/3*p[edges[:, 2], :], (size(edges, 1), :))
                pnew = [p; new_pts]
                np1 = size(pnew, 1)
                # then add 2/3 points
                new_pts = reshape(1/3*p[edges[:, 1], :] + 2/3*p[edges[:, 2], :], (size(edges, 1), :))
                pnew = [pnew; new_pts]
                np2 = size(pnew, 1)
                # finally add center points
                new_pts = reshape(1/3*(p[t[:, 1], :] + p[t[:, 2], :] + p[t[:, 3], :]), (size(t, 1), :))
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
                enew = [e; np0 .+ bndix; np1 .+ bndix]
        else
            error("Unsupported grid order `$order` for dimension `$dim`.")
        end
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
    fmap, faces, bndix = all_faces(t)

1) Find all unique `faces` (nf x 3 array) in the tetrahedral mesh `t`.
2) Determine indices of boundary faces with `bndix`.
3) Map local faces to global faces with `fmap` (nt x 4 array): `fmap[k,i]` is the 
global face number for local face `i` in tetrahedron `k`.
"""
function all_faces(t)
    # form all faces
    if size(t, 2) == 10
        # support for second order tetrahedra
        ftag = [t[:, [1,2,3,5,6,7]]; t[:, [1,2,4,5,9,8]]; t[:, [1,3,4,7,10,8]]; t[:, [2,3,4,6,10,9]]]
        nn = 6
    else
        # otherwise just do corners
        ftag = [t[:, [1,2,3]]; t[:, [1,2,4]]; t[:, [1,3,4]]; t[:, [2,3,4]]]
        nn = 3
    end
    nfaces = size(ftag, 1)

    # sort columns and tag with global indices in last column
    ftag = hcat(sort(ftag, dims=2), 1:nfaces)

    # sort rows
    ftag = sortslices(ftag, dims=1)

    # indices of unique faces
    keep = zeros(Bool, nfaces)
    keep[unique(i -> ftag[i, 1:nn], 1:nfaces)] .= 1

    # keep unique faces
    faces = ftag[keep, 1:nn]

    # mapping from local to global face index
    fmap = cumsum(keep)
    invpermute!(fmap, ftag[:, nn+1])
    fmap = reshape(fmap, :, 4)

    # face `i` has no duplicates if `i` and `i+1` are in `keep`
    bndix = findall(i -> keep[i]*keep[i+1], 1:nfaces-1)
    if keep[nfaces]
        bndix = [bndix; nfaces]
    end
    bndix = cumsum(keep)[bndix]

    return fmap, faces, bndix
end

"""
    emap, edges, bndix = all_edges(t)

1) Find all unique `edges` (ne x 2 array) in the tessellation `t`.
2) Determine indices of boundary edges with `bndix`.
3) Map local edges to global edges with `emap` (nt x dim+1 array): `emap[k,i]` is the 
global edge number for local edge `i` in element `k`.
"""
function all_edges(t)
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
    nedges = size(etag, 1)

    # order node indices so lowest ones are in first column, tag each edge with its global index in 3rd column
    etag = [sort(etag, dims=2)  1:nedges]

    # now sort so that first column goes from lowest to highest node index
    etag = sortslices(etag, dims=1)

    # determine if edge is a duplicate or should be kept
    keep = zeros(Bool, nedges)
    keep[unique(i -> etag[i, 1:2], 1:size(etag, 1))] .= 1

    # only keep unique edges
    edges = etag[keep, 1:2]

    # boundary edges
    if dim == 1 || dim == 2
        # in 1D and 2D, no duplicates
        dup = all(etag[2:end, 1:2] .== etag[1:end-1, 1:2], dims=2)[:]
        dup = [dup; false]
        dup = dup[keep]
        bndix = findall(.!dup)
    elseif dim == 3
        # in 3D, on boundary face
        bfaces = boundary_faces(t)
        _, bedges, _ = all_edges(bfaces)
        bndix = [findfirst(i -> edges[i, :] == bedges[j, :], 1:size(edges, 1)) for j ∈ axes(bedges, 1)]
    end

    # compute mapping to global indices
    emap = cumsum(keep)
    invpermute!(emap, etag[:, 3])
    emap = reshape(emap, :, ne)

    return emap, edges, bndix
end

function boundary_faces(t)
    fmap, faces, bndix = all_faces(t)
    return faces[bndix, :]
end
function boundary_nodes(t)
    if size(t, 2) == 3
        emap, edges, bndix = all_edges(t)
        return unique(edges[bndix, :])
    elseif size(t, 2) == 4
        return unique(boundary_faces(t))
    end
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
        try
            Js[k, :, :] = inv(A)
        catch
            printstyled("Warning: ", color=:red)
            println("Element $k has volume 0.")
        end
    end
    return Jacobians(dets, Js)
end
function Jacobians(gfile::String)
    # get order 1 FE grid
    g = FEGrid(gfile, 1)
    return Jacobians(g)
end