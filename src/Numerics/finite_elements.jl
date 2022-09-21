struct Grid{FM<:AbstractMatrix, IM<:AbstractMatrix, IV<:AbstractVector, IN<:Integer}
    # order of shape functions on this grid
    order::IN

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
    g = Grid(file_name, order)

Construct a grid by loading points `p`, triangles `t`, and boundary nodes `e` from .h5 file.
"""
function Grid(file_name, order::IN) where IN <: Integer
    file = h5open(file_name, "r")
    p = read(file, "p")
    t = read(file, "t")
    e = read(file, "e")
    e = e[:, 1]
    t = convert(Matrix{IN}, t)
    e = convert(Vector{IN}, e)
    if order == 2
        p, t, e = add_midpoints(p, t)
    end
    np = size(p, 1)
    nt = size(t, 1)
    nn = size(t, 2)
    ne = size(e, 1)
    return Grid(order, p, np, t, nt, nn, e, ne)
end

"""
    p = standard_element_nodes(order)

The nodes of a standard element of order `order`.
"""
function standard_element_nodes(order)
    if order == 1
        return [0.0  0.0
                1.0  0.0
                0.0  1.0]
    elseif order == 2
        return [0.0  0.0
                1.0  0.0
                0.0  1.0
                0.5  0.0
                0.5  0.5
                0.0  0.5]
    else
        error("Unsupported shape function order.")
    end
end

"""
    φ(i, ξ; order)

Shape function `i` of order `order` evaluated at the point `ξ`.
"""
function φ(i, ξ; order=1)
    if order == 1
        if i == 1
            return 1 - ξ[1] - ξ[2]
        elseif i == 2
            return ξ[1]
        elseif i == 3
            return ξ[2]
        else
            error("Invalid index for shape function.")
        end
    elseif order == 2
        if i == 1
            return (1 - ξ[1] - ξ[2])*(2*(1 - ξ[1] - ξ[2]) - 1)
        elseif i == 2
            return ξ[1]*(2*ξ[1] - 1)
        elseif i == 3
            return ξ[2]*(2*ξ[2] - 1)
        elseif i == 4
            return 4*ξ[1]*(1 - ξ[1] - ξ[2])
        elseif i == 5
            return 4*ξ[1]*ξ[2]
        elseif i == 6
            return 4*ξ[2]*(1 - ξ[1] - ξ[2])
        else
            error("Invalid index for shape function.")
        end
    else
        error("Unsupported shape function order.")
    end
end

"""
    φ(i, ξ; order)

Derivative of shape function `i` of order `order` in the `j` direction 
evaluated at the point `ξ`.
"""
function ∂φ(i, j, ξ; order=1)
    if order == 1
        if i == 1
            return -1
        elseif i == 2
            if j == 1
                return 1
            elseif j == 2
                return 0
            end
        elseif i == 3
            if j == 1
                return 0
            elseif j == 2
                return 1
            end
        else
            error("Invalid index for shape function.")
        end
    elseif order == 2
        if i == 1
            return 1 - 4*(1 - ξ[1] - ξ[2])
        elseif i == 2
            if j == 1
                return 4*ξ[1] - 1 
            elseif j == 2
                return 0
            end
        elseif i == 3
            if j == 1
                return 0
            elseif j == 2
                return 4*ξ[2] - 1 
            end
        elseif i == 4
            if j == 1
                return 4*(1 - 2*ξ[1] - ξ[2])
            elseif j == 2
                return -4*ξ[1]
            end
        elseif i == 5
            if j == 1
                return 4*ξ[2]
            elseif j == 2
                return 4*ξ[1]
            end
        elseif i == 6
            if j == 1
                return -4*ξ[2]
            elseif j == 2
                return 4*(1 - ξ[1] - 2*ξ[2])
            end
        else
            error("Invalid index for shape function.")
        end
    else
        error("Unsupported shape function order.")
    end
end

struct Jacobians{V<:AbstractVector}
    J::V
    xξ::V
    xη::V
    yξ::V
    yη::V
    ξx::V
    ξy::V
    ηx::V
    ηy::V
end

"""
    J = Jacobians(g)

Compute Jacobian terms for transformations from standard triangle [0 0; 1 0; 0 1] to 
triangles on the grid. Given a triangle on the grid defined by the nodes [x1 y1; x2 y2; x3 y3], 
the transformation ξ ↦ x is defined by
    x = u1 + u2*ξ + u3*η
where
    u1 = [x1 y1]
    u2 = [x2-x1 y2-y1]
    u3 = [x3-x1 y3-y1].
The insverse transform x ↦ ξ is
    ξ = v1 + v2*x + v3*y
where
    v1 = 1/J*[y1*x3-x1*y3, x1*y2-y1*x2)
    v2 = 1/J*[y3-y1,       y1-y2)
    v3 = 1/J*[x1-x3,       x2-x1)
and J = ∂(x, y)/∂(ξ, η) is the jacobian.
"""
function Jacobians(g::Grid)
    # unpack coords
    x = g.p[:, 1]
    y = g.p[:, 2]

    # compute Jacobian terms for each triangle 
    xξ = x[g.t[:, 2]] - x[g.t[:, 1]]
    xη = x[g.t[:, 3]] - x[g.t[:, 1]]
    yξ = y[g.t[:, 2]] - y[g.t[:, 1]]
    yη = y[g.t[:, 3]] - y[g.t[:, 1]]
    J = @. xξ*yη - xη*yξ
    ξx = (y[g.t[:, 3]] - y[g.t[:, 1]])./J
    ξy = (x[g.t[:, 1]] - x[g.t[:, 3]])./J
    ηx = (y[g.t[:, 1]] - y[g.t[:, 2]])./J
    ηy = (x[g.t[:, 2]] - x[g.t[:, 1]])./J
    return Jacobians(J, xξ, xη, yξ, yη, ξx, ξy, ηx, ηy)
end

struct ShapeFunctionIntegrals{M<:AbstractMatrix}
    φφ::M
    φξφ::M
    φηφ::M
    φξφξ::M
    φξφη::M
    φηφξ::M
    φηφη::M
end

"""
    s = ShapeFunctionIntegrals(order)

Compute and store integrals over the standard triangle [0 0; 1 0; 0 1] of the form
    ∫ ∂ₙφⱼ ∂ₘφᵢ dξ
where φᵢ and φⱼ are shape functions of order `order`.
"""
function ShapeFunctionIntegrals(order) 
    # quadrature weights and points
    w, ξ = quad_weights_points(2*order)

    # number of shape functions (div to keep it integer)
    n = div((order + 2)*(order + 1), 2)

    # mass
    φφ = compute_integral_matrix((ξ, i, j) -> φ(j, ξ; order=order)*φ(i, ξ; order=order), w, ξ, n)

    # C
    φξφ = compute_integral_matrix((ξ, i, j) -> ∂φ(j, 1, ξ; order=order)*φ(i, ξ; order=order), w, ξ, n)
    φηφ = compute_integral_matrix((ξ, i, j) -> ∂φ(j, 2, ξ; order=order)*φ(i, ξ; order=order), w, ξ, n)

    # stiffness
    φξφξ = compute_integral_matrix((ξ, i, j) -> ∂φ(j, 1, ξ; order=order)*∂φ(i, 1, ξ; order=order), w, ξ, n)
    φξφη = compute_integral_matrix((ξ, i, j) -> ∂φ(j, 1, ξ; order=order)*∂φ(i, 2, ξ; order=order), w, ξ, n)
    φηφξ = compute_integral_matrix((ξ, i, j) -> ∂φ(j, 2, ξ; order=order)*∂φ(i, 1, ξ; order=order), w, ξ, n)
    φηφη = compute_integral_matrix((ξ, i, j) -> ∂φ(j, 2, ξ; order=order)*∂φ(i, 2, ξ; order=order), w, ξ, n)
    return ShapeFunctionIntegrals(φφ, φξφ, φηφ, φξφξ, φξφη, φηφξ, φηφη)
end

"""
    M = compute_integral_matrix(f, w, ξ, n)

Compute integrals over the standard triangle of the form ∫ f(ξ, i, j) dξ 
for i, j ∈ {1, ..., n}. Quadrature rule defined by weights `w` and integration 
points `ξ`
"""
function compute_integral_matrix(f, w, ξ, n)
    M = zeros(n, n)
    for i=1:n
        for j=1:n
            M[i, j] = std_tri_quad(ξ -> f(ξ, i, j), w, ξ)
        end
    end
    return M
end

"""
    l2 = L2norm(g, s, J, u)

Compute L2 norm, ||u||² ≡ ∫ |u|² dx, of finite element function `u`.
"""
function L2norm(g::Grid, s::ShapeFunctionIntegrals, J::Jacobians, u)
    l2 = 0
    n = size(g.t, 2)
    for k=1:g.nt
        for i=1:n
            for j=1:n
                l2 += u[g.t[k, j]]*u[g.t[k, i]]*s.φφ[i, j]*abs(J.J[k])
            end
        end
    end
    return sqrt(l2)
end

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique `edges` (ne x 2 array) in the triangulation `t`.
Second output is indices to the boundary edges.
Third output `emap` (nt x 3 array) is a mapping from local triangle edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in triangle it.
"""
function all_edges(t)
    # find all edges
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)

    # remove duplicates
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]

    # compute local indices
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)

    # find boundary indices
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
	p2, t2, e2 = add_midpoints(p, t)

Add midpoints to mesh for quadratic functions.
`p2`: N x 2, node coords of original mesh plus new midpoints
`t2`: T x 6, a local-to-global mapping for the T triangle elements
`e2`: E x 1, indices of boundary nodes for original mesh plus new midpoints
"""
function add_midpoints(p, t)
	# Find all the edges at first
    edges, boundary_indices, emap = all_edges(t)

	# Add the midpoints of each edge
    midpts = 1/2 * reshape(p[edges[:, 1], :] + p[edges[:, 2], :], (size(edges, 1), 2))
    p2 = [p; midpts]

	# Add the midpoints of each triangle
    t2 = zeros(size(t, 1), 6)
    t2[:, 1:3] = t
    t2[:, 4:6] = size(p, 1) .+ emap
    t2 = convert(Array{Int64,2}, t2)
    
    # Add the midpoints that were on the boundary
    e2 = [unique(edges[boundary_indices, :][:]); size(p, 1) .+ boundary_indices]
	return p2, t2, e2
end

"""
    t_dict = get_t_dict(p, t)

`t_dict[i]` returns a vector of integer indices for each triangle point `i` is in.
"""
function get_t_dict(p, t)
    t_dict = Dict{Int64, Vector{Int64}}()
    @showprogress "Creating mesh dictionary..." for i=1:size(p, 1)
        t_dict[i] = findall(vec(sum(t .== i, dims=2) .== 1))
    end
    return t_dict
end

"""
    area = tri_area(p)

Compute area of triangle defined by points `p`.
"""
function tri_area(p)
	return 1/2*abs(p[1, 1]*(p[2, 2] - p[3, 2]) + p[2, 1]*(p[3, 2] - p[1, 2]) + p[3, 1]*(p[1, 2] - p[2, 2]))
end

"""
    x = transform_from_std_tri(ξ, p)

Transform point `ξ` defined on standard triangle [0 0; 1 0; 0 1] to x defined on 
triangle with vertices `p`.
"""
function transform_from_std_tri(ξ, p)
    return p[1, :] + (p[2, :] - p[1, :])*ξ[1] + (p[3, :] - p[1, :])*ξ[2]
end

"""
    ξ = transform_to_std_tri(x, p)

Transform point `x` defined on triangle with vertices `p` to standard triangle [0 0; 1 0; 0 1].
"""
function transform_to_std_tri(x, p)
    # jacobian
    J = (p[2, 1] - p[1, 1])*(p[3, 2] - p[1, 2]) - (p[3, 1] - p[1, 1])*(p[2, 2] - p[1, 2])
    v = 1/J * [p[1, 2]*p[3, 1] - p[1, 1]*p[3, 2]  p[1, 1]*p[2, 2] - p[1, 2]*p[2, 1]
               p[3, 2] - p[1, 2]                  p[1, 2] - p[2, 2]
               p[1, 1] - p[3, 1]                  p[2, 1] - p[1, 1]
              ]
    return v[1, :] + v[2, :]*x[1] + v[3, :]*x[2]
end

"""
    bool = pt_in_tri(x, p)

Determine if point `x` is in triangle with nodes `p`.
(See https://stackoverflow.com/a/2049593).
"""
function pt_in_tri(x, p)
    d₁ = pt_sign(x, p[1, :], p[2, :])
    d₂ = pt_sign(x, p[2, :], p[3, :])
    d₃ = pt_sign(x, p[3, :], p[1, :])

    has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
    has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

    return !(has_neg && has_pos)
end
function pt_sign(p₁, p₂, p₃)
    return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
end


"""
    k = get_tri(x, g)

Determine index `k` of triangle on grid `g` in which the point `x` lies.
"""
# function get_tri(x, p, t, t_dict)
#     closest_p = argmin((p[:, 1] .- x[1]).^2 + (p[:, 2] .- x[2]).^2)
#     for k=t_dict[closest_p] # just look at triangles closest_p is in
#         if pt_in_tri(x, p[t[k, 1], :], p[t[k, 2], :], p[t[k, 3], :])
#             return k
#         end
#     end
#     error("Cannot find triangle; p₀=($(x[1]), $(x[2])) is not inside mesh domain.")
# end
function get_tri(x, g::Grid)
    for k=1:g.nt 
        if pt_in_tri(x, g.p[g.t[k, :], :])
            return k
        end
    end
    error("Cannot find triangle; p₀=($(x[1]), $(x[2])) is not inside mesh domain.")
end

# function fem_evaluate(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
#                       t::AbstractArray{<:Integer,2}, t_dict::AbstractDict{IN, Vector{IN}}, 
#                       C₀::AbstractArray{<:Real,3}) where IN <: Integer
#     # find triangle (ξ, η) is in
#     k = get_tri(ξ, η, p, t, t_dict)
    
#     # evaluate there
#     return fem_evaluate(v, ξ, η, p, t, C₀, k)
# end
function fem_evaluate(u, x, g::Grid)
    # find triangle x is in
    k = get_tri(x, g)
    
    # evaluate there
    return fem_evaluate(u, x, g, k)
end
function fem_evaluate(u, x, g::Grid, k)
    # transform to standard triangle
    ξ = transform_to_std_tri(x, g.p[g.t[k, 1:3], :])

    # sum weighted combinations of triangle k's basis functions at x
    return sum([u[g.t[k, i]]*φ(i, ξ; order=g.order) for i=1:g.nn])
end
