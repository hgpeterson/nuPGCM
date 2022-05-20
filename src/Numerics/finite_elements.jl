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
function tri_area(p::AbstractArray{<:Real,2})
    p₁ = p[1, :]
    p₂ = p[2, :]
    p₃ = p[3, :]
	area = 1/2*abs(p₁[1]*(p₂[2] - p₃[2]) + p₂[1]*(p₃[2] - p₁[2]) + p₃[1]*(p₁[2] - p₂[2]))
    return area
end

"""
    C₀ = get_linear_basis_coeffs(p)

Compute coefficients for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
at the nodes defined by 3×2 matrix `p`. C₀[:, i] are the iᵗʰ basis vector coefficients.

If triangles `t` are provided, then C₀ stores coefficients for _all_ bases.
"""
function get_linear_basis_coeffs(p::AbstractArray{<:Real,2})
	V = zeros(3, 3)
	for i=1:3
		V[i, :] = [1 p[i, 1] p[i, 2]]
	end
	return inv(V)
end
function get_linear_basis_coeffs(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2})
    nt = size(t, 1)
    C₀ = zeros(nt, 3, 3)
    for k=1:nt
        C₀[k, :, :] = get_linear_basis_coeffs(p[t[k, :], :])
    end
	return C₀
end

"""
    ϕ = local_basis_func(c, p₀)

Evaluate local basis function defined by coefficients matrix (or vector) `c` at point `p₀` = [ξ η].
"""
function local_basis_func(c::AbstractArray{<:Real}, p₀::AbstractArray{<:Real,1})
    return c'*[1, p₀[1], p₀[2]]
end

# https://stackoverflow.com/a/2049593
function pt_sign(p₁::AbstractArray{<:Real,1}, p₂::AbstractArray{<:Real,1}, p₃::AbstractArray{<:Real,1})
    return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
end
function pt_in_tri(p₀::AbstractArray{<:Real,1}, v₁::AbstractArray{<:Real,1}, v₂::AbstractArray{<:Real,1}, v₃::AbstractArray{<:Real,1})
    d₁ = pt_sign(p₀, v₁, v₂)
    d₂ = pt_sign(p₀, v₂, v₃)
    d₃ = pt_sign(p₀, v₃, v₁)

    has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
    has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

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

function evaluate(u, p₀, p, t, C₀)
    # find triangle p₀ is in
    k₀ = get_tri(p₀, p, t)
    
    # evaluate there
    return evaluate(u, p₀, p, t, C₀, k₀)
end
function evaluate(u, p₀, p, t, C₀, k₀)
    # sum weighted combinations of triangle k₀'s basis functions at p₀
    return u[t[k₀, :]]'*local_basis_func(C₀[k₀, :, :], p₀)
end