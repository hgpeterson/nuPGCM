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
    area = tri_area(p)

Compute area of triangle defined by points `p`.
"""
function tri_area(p::Array{Float64,2})
    p1 = p[1, :]
    p2 = p[2, :]
    p3 = p[3, :]
	area = 1/2*abs(p1[1]*(p2[2] - p3[2]) + p2[1]*(p3[2] - p1[2]) + p3[1]*(p1[2] - p2[2]))
    return area
end

"""
    C₀ = get_linear_basis_coeffs(p)

Compute coefficients for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
at the nodes defined by 3×2 matrix `p`. C₀[:, i] are the iᵗʰ basis vector coefficients.
"""
function get_linear_basis_coeffs(p::Matrix{Float64})
	V = zeros(3, 3)
	for i=1:3
		V[i, :] = [1 p[i, 1] p[i, 2]]
	end
	return inv(V)
end

"""
    ϕ = local_basis_func(c, p₀)

Evaluate local basis function defined by `c` = [c₁ c₂ c₃] at point `p0` = [x y].
"""
function local_basis_func(c::Vector{Float64}, p₀::Vector{Float64})
    return c'*[1, p₀[1], p₀[2]]
end

# https://stackoverflow.com/a/2049593
function pt_sign(p1::Vector{Float64}, p2::Vector{Float64}, p3::Vector{Float64})
    return (p1[1] - p3[1])*(p2[2] - p3[2]) - (p2[1] - p3[1])*(p1[2] - p3[2])
end
function pt_in_tri(pt::Vector{Float64}, v1::Vector{Float64}, v2::Vector{Float64}, v3::Vector{Float64})
    d1 = pt_sign(pt, v1, v2)
    d2 = pt_sign(pt, v2, v3)
    d3 = pt_sign(pt, v3, v1)

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0)
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0)

    return !(has_neg && has_pos)
end
function get_tri(p₀, p, t)
	n_tri = size(t, 1)
    k₀ = 0
    for k=1:n_tri
        if pt_in_tri(p₀, p[t[k, 1], :], p[t[k, 2], :], p[t[k, 3], :])
            k₀ = k
            break
        end
    end
    if k₀ == 0
        error("p₀ is not in mesh domain")
    else
        return k₀
    end
end

function evaluate(u, p₀, p, t)
    # find triangle p₀ is in
    k₀ = get_tri(p₀, p, t)

	# get coeffs for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
    C₀ = get_linear_basis_coeffs(p[t[k₀, :], :])

    # sum weighted combinations of basis functions at p₀
    u₀ = 0
    for i=1:3
        u₀ += u[t[k₀, i]]*local_basis_func(C₀[:, i], p₀)
    end

    return u₀
end
function ∂ξ(u, p₀, p, t)
    # find triangle p₀ is in
    k₀ = get_tri(p₀, p, t)

	# get coeffs for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
    C₀ = get_linear_basis_coeffs(p[t[k₀, :], :])

    # sum weighted combinations of c₂
    return sum(u[t[k₀, :]].*C₀[2, :])
end
function ∂η(u, p₀, p, t)
    # find triangle p₀ is in
    k₀ = get_tri(p₀, p, t)

	# get coeffs for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
    C₀ = get_linear_basis_coeffs(p[t[k₀, :], :])

    # sum weighted combinations of c₃
    return sum(u[t[k₀, :]].*C₀[3, :])
end