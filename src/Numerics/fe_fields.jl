# for defining operations on FEField's
import Base: -
import Base: +
import Base: *
import Base: /
import Base: abs

struct FEField{IN<:Integer,V<:AbstractVector}
    # order of polynomials defining shape functions
    order::IN

    # values of FE field on the nodes of the grid
    values::V

    # grid FE field exists on
    g::FEGrid

    # grid of order 1
    g1::FEGrid
end

"""
    u = FEField(gfile, order, values)

Construct FE field from grid saved at `gfile` of order `order` with node values `values`.
"""
function FEField(gfile::String, order::Integer, values)
    g = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)
    return FEField(order, values, g, g1)
end
function FEField(values, g::FEGrid, g1::FEGrid)
    return FEField(g.order, values, g, g1)
end

# define operations on FEField's
function -(u::FEField, v::FEField)
    return FEField(u.order, u.values - v.values, u.g, u.g1)
end
function +(u::FEField, v::FEField)
    return FEField(u.order, u.values + v.values, u.g, u.g1)
end
function *(u::FEField, v::FEField)
    return FEField(u.order, u.values.*v.values, u.g, u.g1)
end
function /(u::FEField, v::FEField)
    return FEField(u.order, u.values./v.values, u.g, u.g1)
end
function abs(u::FEField)
    return FEField(u.order, abs.(u.values), u.g, u.g1)
end

"""
    l2 = L2norm(u, s, J)

Compute L2 norm, ‖u‖² ≡ ∫ u² dx, of finite element function `u`.
"""
function L2norm(u::FEField, s::ShapeFunctionIntegrals, J::Jacobians)
    L2 = 0
    for k=1:u.g.nt
        for i=1:u.g.nn
            for j=1:u.g.nn
                L2 += u.values[u.g.t[k, j]]*u.values[u.g.t[k, i]]*s.M[i, j]*J.dets[k]
            end
        end
    end
    return sqrt(L2)
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
function get_tri(x, g::FEGrid)
    for k=1:g.nt 
        if pt_in_tri(x, g.p[g.t[k, :], :])
            return k
        end
    end
    error("Cannot find triangle; p₀=($(x[1]), $(x[2])) is not inside mesh domain.")
end

"""
    u(x) = evaluate(u, x)

Evaluate FEField `u` at point `x` on the grid.
"""
function evaluate(u::FEField, x)
    try
        # find triangle x is in
        k = get_tri(x, u.g1)

        # evaluate there
        return evaluate(u, x, k)
    catch
        # if triangle not found, return NaN
        println("p₀=($(x[1]), $(x[2])) outside mesh domain.")
        return NaN
    end
end
function evaluate(u::FEField, x, k)
    # transform to standard triangle
    ξ = transform_to_ref_el(x, u.g1.p[u.g1.t[k, :], :])

    # sum weighted combinations of triangle k's basis functions at x
    return sum([u.values[u.g.t[k, i]]*φ(u.g.s, i, ξ) for i=1:u.g.nn])
end