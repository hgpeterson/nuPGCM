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
    ϕ = local_basis_func(c, p0)

Evaluate local basis function defined by `c` = [c1 c2 c3] at point `p0` = [x y].
"""
function local_basis_func(c, p0)
    return c*[1 p0]'
end