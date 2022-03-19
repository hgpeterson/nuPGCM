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
    ϕ = local_basis_func(c, p₀)

Evaluate local basis function defined by `c` = [c₁ c₂ c₃] at point `p0` = [x y].
"""
function local_basis_func(c, p₀)
    return c*[1 p₀]'
end