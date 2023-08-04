#### Jacobians ####

struct Jacobians{V<:AbstractVector, A<:AbstractArray}
    # |∂x/∂ξ| for each element
    dets::V

    # ∂ξ/∂x for each element
    Js::A
end


"""
    J = Jacobians(el, p, t)

Compute Jacobian terms for transformations from reference element to element on grid.
Given the vertices xᵢ ∈ ℜᵈ of the reference element, the transformation ξ ↦ x is 
    x = A*ξ + b
where
    A for Line = (x₂ - x₁)/2,
    A for Triangle = [x₂-x₁  x₃-x₁
                      y₂-y₁  y₃-y₁],
    A for Wedge = [x₂-x₁  x₃-x₁  0
                   y₂-y₁  y₃-y₁  0
                   0      0      z₄-z₁].
Note that this is only possible for our special wedges that have flat tops and aligned 
bottom and top triangles. We call A = ∂x/∂ξ the Jacobian. To transform from global 
coordinates to the reference element, we then need the inverse of A, J = ∂ξ/∂x:
    J in 1D = ∂ξ/∂x,
    J in 2D = [∂ξ/∂x  ∂ξ/∂y
               ∂η/∂x  ∂η/∂y],
    J in 3D = [∂ξ/∂x  ∂ξ/∂y  ∂ξ/∂z
               ∂η/∂x  ∂η/∂y  ∂η/∂z
               ∂ζ/∂x  ∂ζ/∂y  ∂ζ/∂z].
"""
function Jacobians(el::AbstractElement, p, t)
    # indices
    nt = size(t, 1)

    # pre-allocate
    dets = zeros(nt)
    Js = zeros(nt, el.dim, el.dim)
    
    # loop through elements in grid
    for k=1:nt
        # build A
        A = transformation_matrix(el, p[t[k, :], :])

        # compute determinant
        dets[k] = abs(det(A))

        # invert for J
        Js[k, :, :] .= inv(A)
    end
    return Jacobians(dets, Js)
end

#### Grids ####

struct Grid{E<:AbstractElement, I<:Integer, J<:Jacobians, P<:AbstractVecOrMat, T<:AbstractMatrix, V<:AbstractVector, PT<:AbstractVector}
    # elements on this grid
    el::E

    # Jacobians
    J::J

    # node positions
    p::P

    # number of nodes
    np::I

    # node indices defining each element
    t::T

    # number of elements
    nt::I

    # number of nodes per element
    nn::I

    # edge node indices for different boundaries
    e::Dict{String, V} # dictionary of vectors

    # map from p to t
    p_to_t::PT
end

"""
    g = Grid(order, p, t, e)

Construct a FE grid of order `order` with points `p`, elements `t`, and boundary nodes `e`.
"""
function Grid(el::AbstractElement, p, t, e::Dict)
    # indices
    np = size(p, 1)
    nt = size(t, 1)
    nn = size(t, 2)

    # compute Jacobians
    J = Jacobians(el, p, t)

    # map from p to t
    p_to_t = get_p_to_t(t, np)

    return Grid(el, J, p, np, t, nt, nn, e, p_to_t)
end
function Grid(el::AbstractElement, p, t, e::AbstractVector)
    # make e a dict
    return Grid(el, p, t, Dict("bdy"=>e))
end
function Grid(el::AbstractElement, gfile::String)
    # read grid data
    p, t, e = read_gfile_h5(gfile)
    return Grid(el, p, t, e) 
end
function Grid(el::AbstractElement, g::Grid)
    return Grid(el, g.p, g.t, g.e) 
end

"""
    p, t, e = read_gfile_h5(gfile)

Load grid from HDF5 file `gfile`.
"""
function read_gfile_h5(gfile)
    file = h5open(gfile, "r")
    p::Matrix{Float64} = read(file, "p")
    t::Matrix{Int64} = read(file, "t")
    e::Vector{Int64} = read(file, "e")
    close(file)
    return p, t, e
end

"""
    p_to_t = get_p_to_t(t, np)

Returns a vector of vectors of CartesianIndices `p_to_t` such that p_to_t[i] lists
all the keys in `t` that point to the ith node of the mesh of size `np`.
"""
function get_p_to_t(t, np)
    p_to_t = [CartesianIndex[] for i=1:np]
    for k ∈ axes(t, 1)
        for i ∈ axes(t, 2)
            push!(p_to_t[t[k, i]], CartesianIndex(k, i))
        end
    end
    return p_to_t
end

"""
    g2 = add_midpoints(g)

Add midpoints to the grid `g` to make it second order.
"""
function add_midpoints(g::Grid, el::Triangle)
    # unpack
    np1 = g.np
    p = g.p
    t = g.t
    e = g.e

    # get edges
    emap, edges, bndix = all_edges(t)

    # add midpoints
    mids = 1/2*reshape(p[edges[:, 1], :] + p[edges[:, 2], :], (size(edges, 1), :))
    p2 = [p; mids]
    np2 = size(p2, 1)

    # map to elements
    t2 = hcat(t, np1 .+ emap)

    # add points that were on each boundary of `e`
    e2 = copy(e)
    for bdy ∈ e2
        bdy_name = bdy.first
        bdy_nodes = bdy.second
        for i ∈ bndix
            if edges[i, 1] ∈ bdy_nodes && edges[i, 2] ∈ bdy_nodes
                e2[bdy_name] = [e2[bdy_name]; np1 + i]
            end
        end
    end

    # second order triangle element
    el2 = Triangle(order=2)

    # p2 to t2 map
    p2_to_t2 = get_p_to_t(t2, np2)

    return Grid(el2, g.J, p2, np2, t2, g.nt, el2.n, e2, p2_to_t2)
end
add_midpoints(g::Grid) = add_midpoints(g, g.el)

"""
    emap, edges, bndix = all_edges(t)

1) Find all unique `edges` (ne x 2 array) in the triangluation `t`.
2) Determine indices of boundary edges with `bndix`.
3) Map local edges to global edges with `emap` (nt x 3 array): `emap[k,i]` is the 
global edge number for local edge `i` in element `k`.
"""
function all_edges(t)
    # get all possible edges
    edges = vcat(t[:, [1, 2]], t[:, [2, 3]], t[:, [3, 1]])

    # order node indices so lowest ones are in first column
    sort!(edges, dims=2)

    # tag each edge with its global index in 3rd column
    etag = hcat(edges, 1:3*size(t, 1))

    # now sort so that first column goes from lowest to highest node index
    etag[:] = sortslices(etag, dims=1)

    # determine if edge is a duplicate or should be kept
    dup = all(etag[2:end, 1:2] .== etag[1:end-1, 1:2], dims=2)[:]
    keep = .![false; dup]

    # only keep unique edges
    edges = etag[keep, 1:2]

    # compute mapping to global indices
    emap = cumsum(keep)
    invpermute!(emap, etag[:, 3])
    emap = reshape(emap, :, 3)

    # boundary edges: no duplicates
    dup = [dup; false]
    dup = dup[keep]
    bndix = findall(.!dup)

    return emap, edges, bndix
end

#### Matrices ####

function mass_matrix(g::Grid)
    J = g.J
    el = g.el
    M_el = mass_matrix(el)
    M = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt, i=1:el.n, j=1:el.n
        push!(M, (g.t[k, i], g.t[k, j], J.dets[k]*M_el[i, j]))
    end
    return dropzeros!(sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), g.np, g.np))
end

function stiffness_matrix(g::Grid)
    J = g.J
    el = g.el
    K_el = stiffness_matrix(el)
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        Kᵏ = J.dets[k]*sum(K_el.*JJ, dims=(1, 2))[1, 1, :, :]
        for i=1:el.n, j=1:el.n
            push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np))
end

function stiffness_matrix_zz(g::Grid)
    J = g.J
    el = g.el
    K_el = stiffness_matrix(el)
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        Kᵏ = J.dets[k]*sum(K_el.*JJ, dims=(1, 2))[1, 1, :, :]
        for i=1:el.n, j=1:el.n
            push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np))
end