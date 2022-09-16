struct Grid{FT, IN}
    # node positions
    p::AbstractMatrix{FT}

    # number of nodes
    np::IN

    # node indices defining each element
    t::AbstractMatrix{IN}

    # number of elements
    nt::IN

    # edge node indices
    e::AbstractVector{IN}

    # number of edge nodes
    ne::IN
end

"""
    g = Grid(file_name, order)

Load points `p`, triangles `t`, and boundary nodes `e` from .h5 file.
"""
function Grid(file_name::String, order)
    file = h5open(file_name, "r")
    p = read(file, "p")
    t = read(file, "t")
    e = read(file, "e")
    e = e[:, 1]
    t = convert(Array{Int64,2}, t)
    e = convert(Array{Int64,1}, e)
    if order == 2
        p, t, e = add_midpoints(p, t)
    end
    np = size(p, 1)
    nt = size(t, 1)
    ne = size(e, 1)
    return Grid(p, np, t, nt, e, ne)
end

struct StandardElement{FT, IN}
    # quadrature weights and points
    int_wts::AbstractVector{FT}
    int_pts::AbstractMatrix{FT}

    # number of integration points
    n_int_pts::IN

    # node positions
    p::AbstractMatrix{FT}

    # number of nodes
    n_el_nodes::IN

    # shape functions and their derivatives evaluated at integration points
    φ_int_pts::AbstractMatrix{FT}
    ∂φ_int_pts::AbstractArray{FT, 3}
end

function StandardElement(order, degree)
    # get quadrature weights and points
    w, ξ = quad_weights_points(degree)
    nξ = size(ξ, 1)

    # get nodes on standard element
    p = standard_element_nodes(order)
    np = size(p, 1)

    # evaluate shape functions and their derivatives at the integration points
    φ_int_pts = zeros(np, nξ)
    ∂φ_int_pts = zeros(np, 2, nξ)
    for i=1:np
        for j=1:nξ
            φ_int_pts[i, j] = φ(i, ξ[j, :]; order=order)
            for k=1:2
                ∂φ_int_pts[i, k, j] = ∂φ(i, k, ξ[j, :]; order=order)
            end
        end
    end

    return StandardElement(w, ξ, nξ, p, np, φ_int_pts, ∂φ_int_pts)
end

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

struct FESpace{FT, IN}
    # order of shape functions
    order::IN

    # degree of integration
    degree::IN

    # grid
    grid::Grid

    # standard element
    std_el::StandardElement

    # Jacobian ∂(x, y)/∂(ξ, η) at each integration point
    J_int_pts::AbstractMatrix{FT}

    # shape function derivatives ∂φ∂x(ξ, η) and ∂φ∂y(ξ, η) at each integration point
    ∂φ_int_pts::AbstractArray{FT, 4}
end

function FESpace(fname, order, degree)
    # load grid
    g = Grid(fname, order)

    # define standard element
    s = StandardElement(order, degree)

    # compute Jacobians and derivatives
    ∂φ = zeros(g.nt, s.n_el_nodes, 2, s.n_int_pts)
    J = zeros(g.nt, s.n_int_pts)
    for k=1:g.nt
        for i=1:s.n_int_pts
            # compute Jacobian ∂(x, y)/∂(ξ, η)
            x_ξ = dot(s.∂φ_int_pts[:, 1, i], g.p[g.t[k, :], 1])
            x_η = dot(s.∂φ_int_pts[:, 2, i], g.p[g.t[k, :], 1])
            y_ξ = dot(s.∂φ_int_pts[:, 1, i], g.p[g.t[k, :], 2])
            y_η = dot(s.∂φ_int_pts[:, 2, i], g.p[g.t[k, :], 2])
            J[k, i] = x_ξ*y_η - x_η*y_ξ
            for j=1:s.n_el_nodes
                # compute shape function derivatives ∂φ∂x(ξ, η) and ∂φ∂y(ξ, η)
                ∂φ[k, j, 1, i] =  1/J[k, i]*(s.∂φ_int_pts[j, 1, i]*y_η - s.∂φ_int_pts[j, 2, i]*y_ξ)
                ∂φ[k, j, 2, i] = -1/J[k, i]*(s.∂φ_int_pts[j, 1, i]*x_η - s.∂φ_int_pts[j, 2, i]*x_ξ)
            end
        end
    end
    return FESpace(order, degree, g, s, J, ∂φ)
end

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique `edges` (ne x 2 array) in the triangulation `t`.
Second output is indices to the boundary edges.
Third output `emap` (nt x 3 array) is a mapping from local triangle edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in triangle it.
"""
function all_edges(t::AbstractArray{<:Integer,2})
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
function add_midpoints(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2})
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
function tri_area(p::AbstractArray{<:Real,2})
	return 1/2*abs(p[1, 1]*(p[2, 2] - p[3, 2]) + p[2, 1]*(p[3, 2] - p[1, 2]) + p[3, 1]*(p[1, 2] - p[2, 2]))
end

"""
    x = transform_from_std_tri(x̃, p)

Transform point `x̃` defined on standard triangle [0 0; 1 0; 0 1] to x defined on 
triangle with vertices `p`.
"""
function transform_from_std_tri(x̃::AbstractVector{<:Real}, p::AbstractMatrix{<:Real})
    return p[1, :] + (p[2, :] - p[1, :])*x̃[1] + (p[3, :] - p[1, :])*x̃[2]
end

"""
    x̃ = transform_to_std_tri(x, p)

Transform point `x` defined on triangle with vertices `p` to standard triangle [0 0; 1 0; 0 1].
"""
function transform_to_std_tri(x::AbstractVector{<:Real}, p::AbstractMatrix{<:Real})
    D = -p[1, 2]*p[2, 1] + p[1, 1]*p[2, 2] + p[1, 2]*p[3, 1] - p[2, 2]*p[3, 1] - p[1, 1]*p[3, 2] + p[2, 1]*p[3, 2]
    v = 1/D * [p[1, 2]*p[3, 1] - p[1, 1]*p[3, 2]  p[1, 1]*p[2, 2] - p[1, 2]*p[2, 1]
               p[3, 2] - p[1, 2]                  p[1, 2] - p[2, 2]
               p[1, 1] - p[3, 1]                  p[2, 1] - p[1, 1]
              ]
    return v[1, :] + v[2, :]*x[1] + v[3, :]*x[2]
end

"""
    C₀ = get_shape_func_coeffs(p)

Compute coefficients C₀ for linear or quadratic shape functions 
at the nodes defined by `p`. C₀[i, :] are the iᵗʰ basis vector coefficients.

If triangles `t` are provided, then C₀ stores coefficients for _all_ bases.
"""
function get_shape_func_coeffs(p::AbstractArray{<:Real,2})
    n = size(p, 1)
	V = zeros(n, n)
	for i=1:n
	    ξ = p[i, 1]
	    η = p[i, 2]
        if n == 3
		    V[:, i] = [1, ξ, η]
        elseif n == 6
	        V[:, i] = [1, ξ, η, ξ^2, ξ*η, η^2]
        else
            error("Unsupported shape function order.")
        end
	end
	return inv(V)
end
function get_shape_func_coeffs(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2})
    nt = size(t, 1)
    C₀ = zeros(nt, size(t, 2), size(t, 2))
    for k=1:nt
        C₀[k, :, :] = get_shape_func_coeffs(p[t[k, :], :])
    end
	return C₀
end

"""
    φ = shape_func(c, ξ, η)

Evaluate shape function defined by coefficients matrix (or vector) `c` at point (ξ, η).
"""
function shape_func(c::AbstractArray{<:Real,1}, ξ::Real, η::Real; dξ=0, dη=0)
    if size(c, 1) == 3
        # first order = linear
        if dξ == 0 && dη == 0
            return c[1] + c[2]*ξ + c[3]*η
        elseif dξ == 1 && dη == 0
            return c[2]
        elseif dξ == 0 && dη == 1
            return c[3]
        else
            error("Unsupported derivatives of linear shape function: dξ = $dξ, dη = $dη.")
        end
    elseif size(c, 1) == 6
        # second order = quadratic
        if dξ == 0 && dη == 0
            return c[1] + c[2]*ξ + c[3]*η + c[4]*ξ^2 + c[5]*ξ*η + c[6]*η^2 
        elseif dξ == 1 && dη == 0
            return c[2] + 2*c[4]*ξ + c[5]*η
        elseif dξ == 0 && dη == 1
            return c[3] + c[5]*ξ + 2*c[6]*η
        elseif dξ == 2 && dη == 0
            return 2*c[4]
        elseif dξ == 0 && dη == 2
            return 2*c[6]
        else
            error("Unsupported derivatives of quadratic shape function: dξ = $dξ, dη = $dη.")
        end
    else
        error("Invalid coefficient vector. Only first and second degree polynomials supported.")
    end
end
function shape_func(C::AbstractArray{<:Real,2}, ξ::Real, η::Real; dξ=0, dη=0)
    n = size(C, 1)
    v = zeros(n)
    for i=1:n
        v[i] = shape_func(C[i, :], ξ, η; dξ=dξ, dη=dη)
    end
    return v
end

# https://stackoverflow.com/a/2049593
function pt_sign(p₁::AbstractArray{<:Real,1}, p₂::AbstractArray{<:Real,1}, p₃::AbstractArray{<:Real,1})
    return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
end
function pt_in_tri(ξ::Real, η::Real, v₁::AbstractArray{<:Real,1}, v₂::AbstractArray{<:Real,1}, v₃::AbstractArray{<:Real,1})
    d₁ = pt_sign([ξ, η], v₁, v₂)
    d₂ = pt_sign([ξ, η], v₂, v₃)
    d₃ = pt_sign([ξ, η], v₃, v₁)

    has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
    has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

    return !(has_neg && has_pos)
end
function get_tri(ξ::Real, η::Real, p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer, 2}, 
                 t_dict::AbstractDict{IN, Vector{IN}}) where IN <: Integer
    closest_p = argmin((p[:, 1] .- ξ).^2 + (p[:, 2] .- η).^2)
    for k=t_dict[closest_p] # just look at triangles closest_p is in
        if pt_in_tri(ξ, η, p[t[k, 1], :], p[t[k, 2], :], p[t[k, 3], :])
            return k
        end
    end
    error("Cannot find triangle; p₀=($ξ, $η) is not inside mesh domain.")
end
function get_tri(ξ::Real, η::Real, p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer, 2})
    for k=1:size(t, 1) 
        if pt_in_tri(ξ, η, p[t[k, 1], :], p[t[k, 2], :], p[t[k, 3], :])
            return k
        end
    end
    error("Cannot find triangle; p₀=($ξ, $η) is not inside mesh domain.")
end

function fem_evaluate(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
                      t::AbstractArray{<:Integer,2}, t_dict::AbstractDict{IN, Vector{IN}}, 
                      C₀::AbstractArray{<:Real,3}) where IN <: Integer
    # find triangle (ξ, η) is in
    k = get_tri(ξ, η, p, t, t_dict)
    
    # evaluate there
    return fem_evaluate(v, ξ, η, p, t, C₀, k)
end
function fem_evaluate(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
                      t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # find triangle (ξ, η) is in
    k = get_tri(ξ, η, p, t)
    
    # evaluate there
    return fem_evaluate(v, ξ, η, p, t, C₀, k)
end
function fem_evaluate(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
                      t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3}, k::Integer)
    # sum weighted combinations of triangle k's basis functions at p₀
    return v[t[k, :]]'*shape_func(C₀[k, :, :], ξ, η)
end
