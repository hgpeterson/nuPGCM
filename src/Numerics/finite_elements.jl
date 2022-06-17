"""
    p, t, e = load_mesh(file_name)

Load points `p`, triangles `t`, and boundary nodes `e` from .h5 file.
"""
function load_mesh(file_name::String)
    file = h5open(file_name, "r")
    p = read(file, "p")
    t = read(file, "t")
    e = read(file, "e")
    e = e[:, 1] # only need first column to make it nodes
    # convert t and e to integer arrays
    t = convert(Array{Int64,2}, t)
    e = convert(Array{Int64,1}, e)
    return p, t, e
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
    t2 = convert.(Int64, t2)
    
    # Add the midpoints that were on the boundary
    e2 = [unique(edges[boundary_indices, :][:]); size(p, 1) .+ boundary_indices]
	return p2, t2, e2
end

"""
    area = tri_area(p)

Compute area of triangle defined by points `p`.
"""
function tri_area(p::AbstractArray{<:Real,2})
	return 1/2*abs(p[1, 1]*(p[2, 2] - p[3, 2]) + p[2, 1]*(p[3, 2] - p[1, 2]) + p[3, 1]*(p[1, 2] - p[2, 2]))
end

"""
    C₀ = get_shape_func_coeffs(p)

Compute coefficients for linear and quadratic shape functions C₀ matrix
at the nodes defined by `p`. C₀[:, i] are the iᵗʰ basis vector coefficients.

If triangles `t` are provided, then C₀ stores coefficients for _all_ bases.
"""
function get_shape_func_coeffs(p::AbstractArray{<:Real,2})
    n = size(p, 1)
	V = zeros(n, n)
	for i=1:n
	    ξ = p[i, 1]
	    η = p[i, 2]
        if n == 3
		    V[i, :] = [1 ξ η]
        elseif n == 6
	        V[i, :] = [1 ξ η ξ^2 η^2 ξ*η]
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
    ϕ = shape_func(c, ξ₀, η₀)

Evaluate shape function defined by coefficients matrix (or vector) `c` at point (ξ₀, η₀).
"""
function shape_func(c::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real; dξ=0, dη=0)
    if size(c, 1) == 3
        # first order = linear
        if dξ == 0 && dη == 0
            return c[1] + c[2]*ξ₀ + c[3]*η₀
        elseif dξ == 1 && dη == 0
            return c[2]
        elseif dξ == 0 && dη == 1
            return c[3]
        else
            error("Unsupported derivatives of linear shape function: dξ = $dξ, dη = $dη.")
        end
    elseif size(c, 1) == 6
        # second order = quadratic
        return c[1] + c[2]*ξ₀ + c[3]*η₀ + c[4]*ξ₀^2 + c[5]*η₀^2 + c[6]*ξ₀*η₀
        if dξ == 0 && dη == 0
            return c[1] + c[2]*ξ₀ + c[3]*η₀ + c[4]*ξ₀^2 + c[5]*η₀^2 + c[6]*ξ₀*η₀
        elseif dξ == 1 && dη == 0
            return c[2] + 2*c[4]*ξ₀ + c[6]*η₀
        elseif dξ == 0 && dη == 1
            return c[3] + 2*c[5]*η₀ + c[6]*ξ₀
        else
            error("Unsupported derivatives of quadratic shape function: dξ = $dξ, dη = $dη.")
        end
    else
        error("Invalid coefficient vector. Only first and second degree polynomials supported.")
    end
end
function shape_func(c::AbstractArray{<:Real,2}, ξ₀::Real, η₀::Real; dξ=0, dη=0)
    n = size(c, 1)
    v = zeros(n)
    for i=1:n
        v[i] = shape_func(c[:, i], ξ₀, η₀; dξ=dξ, dη=dη)
    end
    return v
end

# https://stackoverflow.com/a/2049593
function pt_sign(p₁::AbstractArray{<:Real,1}, p₂::AbstractArray{<:Real,1}, p₃::AbstractArray{<:Real,1})
    return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
end
function pt_in_tri(ξ₀::Real, η₀::Real, v₁::AbstractArray{<:Real,1}, v₂::AbstractArray{<:Real,1}, v₃::AbstractArray{<:Real,1})
    d₁ = pt_sign([ξ₀, η₀], v₁, v₂)
    d₂ = pt_sign([ξ₀, η₀], v₂, v₃)
    d₃ = pt_sign([ξ₀, η₀], v₃, v₁)

    has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
    has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

    return !(has_neg && has_pos)
end
function get_tri(ξ₀::Real, η₀::Real, p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2})
    closest_p = argmin((p[:, 1] .- ξ₀).^2 + (p[:, 2] .- η₀).^2)
    for k=1:size(t, 1)
        if closest_p in t[k, :]
            if pt_in_tri(ξ₀, η₀, p[t[k, 1], :], p[t[k, 2], :], p[t[k, 3], :])
                return k
            end
        end
    end
    error("Cannot find triangle; p₀=($ξ₀, $η₀) is not inside mesh domain.")
end

function fem_evaluate(v::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real, p::AbstractArray{<:Real,2}, 
                      t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # find triangle p₀ is in
    k₀ = get_tri(ξ₀, η₀, p, t)
    
    # evaluate there
    return fem_evaluate(v, ξ₀, η₀, p, t, C₀, k₀)
end
function fem_evaluate(v::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real, p::AbstractArray{<:Real,2}, 
                      t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3}, k₀::Integer)
    # sum weighted combinations of triangle k₀'s basis functions at p₀
    return v[t[k₀, :]]'*shape_func(C₀[k₀, :, :], ξ₀, η₀)
end