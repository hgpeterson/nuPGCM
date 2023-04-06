# for defining operations on FEField's
import Base: -, +, *, /, ^, log, abs, maximum, minimum, argmax, argmin, getindex

struct FEField{IN<:Integer,V<:AbstractVector}
    # order of polynomials defining shape functions
    order::IN

    # values of FE field on the nodes of the grid
    values::V

    # grid FE field exists on
    g::FEGrid
end

"""
    u = FEField(gfile, order, values)

Construct FE field from grid saved at `gfile` of order `order` with node values `values`.
"""
function FEField(gfile::String, order::Integer, values)
    g = FEGrid(gfile, order)
    return FEField(order, values, g)
end
function FEField(values::AbstractArray, g::FEGrid)
    return FEField(g.order, values, g)
end
function FEField(value::Number, g::FEGrid)
    return FEField(value*ones(g.np), g)
end
function FEField(f::Function, g::FEGrid)
    return FEField([f(g.p[i, :]) for i=1:g.np], g)
end

# define operations on FEField's
function -(u::FEField, v::FEField)
    return FEField(u.order, u.values - v.values, u.g)
end
function -(u::FEField)
    return FEField(u.order, -u.values, u.g)
end
function +(u::FEField, v::FEField)
    return FEField(u.order, u.values + v.values, u.g)
end
function *(u::FEField, v::FEField)
    return FEField(u.order, u.values.*v.values, u.g)
end
function *(u::FEField, c)
    return FEField(u.order, u.values*c, u.g)
end
function /(u::FEField, v::FEField)
    return FEField(u.order, u.values./v.values, u.g)
end
function ^(u::FEField, n)
    return FEField(u.order, u.values.^n, u.g)
end
function log(u::FEField)
    return FEField(u.order, log.(u.values), u.g)
end
function abs(u::FEField)
    return FEField(u.order, abs.(u.values), u.g)
end
function maximum(u::FEField)
    return maximum(u.values)
end
function minimum(u::FEField)
    return minimum(u.values)
end
function argmax(u::FEField)
    return argmax(u.values)
end
function argmin(u::FEField)
    return argmin(u.values)
end
function getindex(u::FEField, I)
    return u.values[I]
end

"""
    l2 = L2norm(u)

Compute L2 norm, ‖u‖ ≡ √(∫ u² dx), of finite element function `u`.
"""
function L2norm(u::FEField)
    return sqrt(sum(u.values[u.g.t[k, j]]*u.values[u.g.t[k, i]]*u.g.sfi.M[i, j]*u.g.J.dets[k] for k=1:u.g.nt, i=1:u.g.nn, j=1:u.g.nn))
end

"""
    bool = pt_in_line(x, p)

Determine if point `x` is in line segment with nodes `p`.
"""
function pt_in_line(x, p)
    return minimum(p) ≤ x ≤ maximum(p)
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
    bool = pt_in_tet(x, p)

Determine if point `x` is in tetrahedron with nodes `p`.
(See https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not).
"""
function pt_in_tet(x, p)
    v1 = p[1, :]
    v2 = p[2, :]
    v3 = p[3, :]
    v4 = p[4, :]
    return same_side(x, v1, v2, v3, v4) &&
           same_side(x, v2, v3, v4, v1) &&
           same_side(x, v3, v4, v1, v2) &&
           same_side(x, v4, v1, v2, v3)
end
function same_side(x, v1, v2, v3, v4)
    normal = cross(v2 - v1, v3 - v1)
    dotv4 = dot(normal, v4 - v1)
    dotx = dot(normal, x - v1)
    return (sign(dotv4) == sign(dotx)) || dotx == 0
end


"""
    k = get_k(x, g)

Determine index `k` of element on grid `g` in which the point `x` lies.
"""
function get_k(x, g)
    if g.dim == 1
        pt_check = pt_in_line
    elseif g.dim == 2
        pt_check = pt_in_tri
    elseif g.dim == 3
        pt_check = pt_in_tet
    else
        error("Dimension $dim not supported for evaluation.")
    end
    for k=1:g.nt 
        if pt_check(x, g.p[g.t[k, 1:g.dim+1], :])
            return k
        end
    end
    error("Cannot find element; p₀=$x is not inside mesh domain.")
end

"""
    u(x)

Evaluate FEField `u` at point `x` on the grid.
"""
function (u::FEField)(x)
    try
        # find element x is in
        k = get_k(x, u.g)

        # evaluate there
        return u(x, k)
    catch
        return NaN
    end
end
function (u::FEField)(x, k)
    # transform to reference element
    ξ = transform_to_ref_el(x, u.g.p[u.g.t[k, 1:u.g.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u.values[u.g.t[k, i]]*φ(u.g.sf, i, ξ) for i=1:u.g.nn)
end

"""
    ∂(u, x, j)

Evaluate the `j`th partial derivative of `u` at `x`.
"""
function ∂(u::FEField, x, j)
    # try
        # find element x is in
        k = get_k(x, u.g)

        # evaluate there
        return ∂(u, x, k, j)
    # catch
    #     return NaN
    # end
end
function ∂(u::FEField, x, k, j)
    # transform to reference element
    ξ = transform_to_ref_el(x, u.g.p[u.g.t[k, 1:u.g.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u.values[u.g.t[k, i]]*∂φ(u.g.sf, i, l, ξ)*u.g.J.Js[k, l, j] for i=1:u.g.nn, l=1:u.g.dim)
end

# shortcuts
∂x(u::FEField, x) = ∂(u, x, 1)
∂y(u::FEField, x) = ∂(u, x, 2)
∂z(u::FEField, x) = ∂(u, x, 3)
∂x(u::FEField, x, k) = ∂(u, x, k, 1)
∂y(u::FEField, x, k) = ∂(u, x, k, 2)
∂z(u::FEField, x, k) = ∂(u, x, k, 3)