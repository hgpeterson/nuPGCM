# for defining operations on FEField's
import Base: -, +, *, /, ^, log, abs, maximum, minimum, argmax, argmin, getindex

abstract type Field end

# operations on Fields
maximum(u::Field) = maximum(u.values)
minimum(u::Field) = minimum(u.values)
argmax(u::Field) = argmax(u.values)
argmin(u::Field) = argmin(u.values)

struct FEField{V<:AbstractVector, G<:Grid} <: Field
    # values of FE field on the nodes of the grid
    values::V

    # grid field exists on
    g::G
end

"""
    u = FEField(values, g)
    u = FEField(value, g)
    u = FEField(f, g)

Construct finite element field of `values` over a grid `g`. If only one `value` is given, 
field is constant. If a function is given, `values` are `f` at the nodes of `g`.
"""
function FEField(value::Number, g::Grid)
    return FEField(value*ones(g.np), g)
end
function FEField(f::Function, g::Grid)
    return FEField([f(g.p[i, :]) for i=1:g.np], g)
end

# operations on FEFields
getindex(u::FEField, i) = u.values[i]
^(u::FEField, n::Number) = FEField(u.values .^ n, u.g)
log(u::FEField) = FEField(log.(u.values), u.g)
abs(u::FEField) = FEField(abs.(u.values), u.g)
-(u::FEField, v::FEField) = FEField(u.values - v.values, u.g)
-(u::FEField) = FEField(-u.values, u.g)
+(u::FEField, v::FEField) = FEField(u.values + v.values, u.g)
*(u::FEField, v::FEField) = FEField(u.values .* v.values, u.g)
/(u::FEField, v::FEField) = FEField(u.values ./ v.values, u.g)
/(u::FEField, n::Number) = FEField(u.values / n, u.g)

struct DGField{M<:AbstractMatrix, G<:Grid} <: Field
    # values of DG field on the nodes of the grid
    values::M

    # grid field exists on
    g::G
end

"""
    u = DGField(values, g)
    u = DGField(value, g)
    u = DGField(f, g)

Construct discontinuous Galerkin field of `values` over a grid `g`. If only one `value` is given, 
field is constant. If a function is given, `values` are `f` at the nodes of `g`.
"""
function DGField(value::Number, g::Grid)
    return DGField(value*ones(g.nt, g.nn), g)
end
function DGField(f::Function, g::Grid)
    return DGField([f(g.p[g.t[k, i], :]) for k=1:g.nt, i=1:g.nn], g)
end

# operations on DGFields
getindex(u::DGField, i, j) = u.values[i, j]
^(u::DGField, n::Number) = DGField(u.values .^ n, u.g)
log(u::DGField) = DGField(log.(u.values), u.g)
abs(u::DGField) = DGField(abs.(u.values), u.g)
-(u::DGField, v::DGField) = DGField(u.values - v.values, u.g)
-(u::DGField) = DGField(-u.values, u.g)
+(u::DGField, v::DGField) = DGField(u.values + v.values, u.g)
*(u::DGField, v::DGField) = DGField(u.values .* v.values, u.g)
/(u::DGField, v::DGField) = DGField(u.values ./ v.values, u.g)

# operations between FE and DG fields
*(u::DGField, v::FEField) = DGField([u[k, i]*v[u.g.t[k, i]] for k=1:u.g.nt, i=1:u.g.nn], u.g)
*(u::FEField, v::DGField) = v * u
/(u::DGField, v::FEField) = DGField([u[k, i]/v[u.g.t[k, i]] for k=1:u.g.nt, i=1:u.g.nn], u.g)
/(u::FEField, v::DGField) = (v / u)^-1

# convert DGField to FEField by averaging
function FEField(u::DGField) 
    g = u.g
    u_cg = zeros(g.np)
    count = zeros(Int64, g.np) # number of triangles per node
    for k=1:g.nt
        for i=1:g.nn
            u_cg[g.t[k, i]] += u[k, i]
            count[g.t[k, i]] += 1
        end
    end
    u_cg ./= count
    return FEField(u_cg, g)
end

struct FVField{V<:AbstractVector, G<:Grid} <: Field
    # values of FV field on the elements of the grid
    values::V

    # grid field exists on
    g::G
end

"""
    u = FVField(values, g)
    u = FVField(value, g)
    u = FVField(f, g)

Construct finite volume field of `values` over a grid `g`. If only one `value` is given, 
field is constant. If a function is given, `values` are `f` on the elements of `g`.
"""
function FVField(value::Number, g::Grid)
    return FVField(value*ones(g.nt), g)
end
function FVField(f::Function, g::Grid)
    return FVField([f(sum(g.p[g.t[k, i], :] for i=1:g.nn)/g.nn) for k=1:g.nt], g)
end

# operations on FVFields
getindex(u::FVField, k) = u.values[k]
^(u::FVField, n::Number) = FVField(u.values .^ n, u.g)
log(u::FVField) = FVField(log.(u.values), u.g)
abs(u::FVField) = FVField(abs.(u.values), u.g)
-(u::FVField, v::FVField) = FVField(u.values - v.values, u.g)
-(u::FVField) = FVField(-u.values, u.g)
+(u::FVField, v::FVField) = FVField(u.values + v.values, u.g)
*(u::FVField, v::FVField) = FVField(u.values .* v.values, u.g)
/(u::FVField, v::FVField) = FVField(u.values ./ v.values, u.g)

# convert DGField to FVField by averaging
FVField(u::DGField) = FVField([sum(u[k, i] for i=1:u.g.nn)/u.g.nn for k=1:u.g.nt], u.g)

"""
    l2 = L2norm(u)

Compute L2 norm, ‖u‖ ≡ √(∫ u² dx), of finite element function `u`.
"""
function L2norm(u::FEField)
    return sqrt(sum(u[u.g.t[k, j]]*u[u.g.t[k, i]]*u.g.sfi.M[i, j]*u.g.J.dets[k] for k=1:u.g.nt, i=1:u.g.nn, j=1:u.g.nn))
end
function L2norm(u::DGField)
    return sqrt(sum(u[k, j]*u[k, i]*u.g.sfi.M[i, j]*u.g.J.dets[k] for k=1:u.g.nt, i=1:u.g.nn, j=1:u.g.nn))
end
function L2norm(u::FVField)
    return sqrt(sum(u[k]^2*u.g.J.dets[k] for k=1:u.g.nt))
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
    bool = pt_in_wedge(x, p)

Determine if point `x` is in wedge with nodes `p` (assuming wedge is 
aligned in the vertical).
"""
function pt_in_wedge(x, p)
    return pt_in_tri(x[1:2], p[1:3, 1:2]) && pt_in_line(x[3], p[[1,4], 3])
end


"""
    k = get_k(x, g)

Determine index `k` of element on grid `g` in which the point `x` lies.
"""
function get_k(x, g, pt_check::Function)
    for k=1:g.nt 
        if pt_check(x, g.p[g.t[k, 1:g.el.dim+1], :])
            return k
        end
    end
    error("Cannot find element; p₀=$x is not inside mesh domain.")
end
function get_k(x, g, el::Line)
    return get_k(x, g, pt_in_line)
end
function get_k(x, g, el::Triangle)
    return get_k(x, g, pt_in_tri)
end
function get_k(x, g, el::Wedge)
    return get_k(x, g, pt_in_wedge)
end
function get_k(x, g)
    return get_k(x, g, g.el)
end

"""
    u(x)

Evaluate FEField `u` at point `x` on the grid.
"""
function (u::Field)(x)
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
    ξ₀ = ξ(u.g.el, x, u.g.p[u.g.t[k, 1:u.g.el.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[u.g.t[k, i]]*φ(u.g.el, ξ₀, i) for i=1:u.g.nn)
end
function (u::DGField)(x, k)
    # transform to reference element
    ξ₀ = ξ(u.g.el, x, u.g.p[u.g.t[k, 1:u.g.el.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[k, i]*φ(u.g.el, ξ₀, i) for i=1:u.g.nn)
end
function (u::FVField)(x, k)
    return u[k]
end

"""
    ∂(u, x, j)

Evaluate the `j`th partial derivative of `u` at `x`.
"""
function ∂(u::Field, x, j)
    try
        # find element x is in
        k = get_k(x, u.g)

        # evaluate there
        return ∂(u, x, k, j)
    catch
        return NaN
    end
end
function ∂(u::FEField, x, k, j)
    # transform to reference element
    ξ₀ = ξ(u.g.el, x, u.g.p[u.g.t[k, 1:u.g.el.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[u.g.t[k, i]]*∂φ(u.g.el, ξ₀, i, l)*u.g.J.Js[k, l, j] for i=1:u.g.nn, l=1:u.g.dim)
end
function ∂(u::DGField, x, k, j)
    # transform to reference element
    ξ₀ = ξ(u.g.el, x, u.g.p[u.g.t[k, 1:u.g.el.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[k, i]*∂φ(u.g.el, ξ₀, i, l)*u.g.J.Js[k, l, j] for i=1:u.g.nn, l=1:u.g.dim)
end
function ∂(u::FVField, x, k, j)
    return 0
end

# shortcuts
∂x(u::Field, x) = ∂(u, x, 1)
∂y(u::Field, x) = ∂(u, x, 2)
∂z(u::Field, x) = ∂(u, x, 3)
∂x(u::Field, x, k) = ∂(u, x, k, 1)
∂y(u::Field, x, k) = ∂(u, x, k, 2)
∂z(u::Field, x, k) = ∂(u, x, k, 3)