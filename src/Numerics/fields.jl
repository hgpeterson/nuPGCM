# for defining operations on FEField's
import Base: -, +, *, /, ^, log, abs, maximum, minimum, argmax, argmin, getindex

abstract type Field end

# operations on Fields
^(u::Field, n::Number) = FEField(u.order, u.values .^ n, u.g)
log(u::Field) = FEField(u.order, log.(u.values), u.g)
abs(u::Field) = FEField(u.order, abs.(u.values), u.g)
maximum(u::Field) = maximum(u.values)
minimum(u::Field) = minimum(u.values)
argmax(u::Field) = argmax(u.values)
argmin(u::Field) = argmin(u.values)

struct FEField{IN<:Integer,V<:AbstractVector} <: Field
    # order of polynomials defining shape functions
    order::IN

    # values of FE field on the nodes of the grid
    values::V

    # grid FE field exists on
    g::Grid
end

"""
    u = FEField(gfile, order, values)

Construct FE field from grid saved at `gfile` of order `order` with node values `values`.
"""
function FEField(order::Integer, values::AbstractVector, gfile::String)
    g = Grid(gfile, order)
    return FEField(order, values, g)
end
function FEField(values::AbstractVector, g::Grid)
    return FEField(g.order, values, g)
end
function FEField(value::Number, g::Grid)
    return FEField(value*ones(g.np), g)
end
function FEField(f::Function, g::Grid)
    return FEField([f(g.p[i, :]) for i=1:g.np], g)
end

# operations on FEFields
getindex(u::FEField, i) = u.values[i]

-(u::FEField, v::AbstractVector) = FEField(u.order, u.values - v, u.g)
-(u::FEField) = FEField(u.order, -u.values, u.g)
-(u::AbstractVector, v::FEField) = -(v - u)
-(u::FEField, v::FEField) = u - v.values

+(u::FEField, v::AbstractVector) = FEField(u.order, u.values + v, u.g)
+(u::AbstractVector, v::FEField) = v + u
+(u::FEField, v::FEField) = u + v.values

*(u::FEField, v::AbstractVector) = FEField(u.order, u.values .* v, u.g)
*(u::AbstractVector, v::FEField) = v * u
*(u::FEField, v::FEField) = u * v.values

/(u::FEField, v::AbstractVector) = FEField(u.order, u.values ./ v, u.g)
/(u::AbstractVector, v::FEField) = v / u
/(u::FEField, v::FEField) = u / v.values

struct DGField{IN<:Integer,M<:AbstractMatrix} <: Field
    # order of polynomials defining shape functions
    order::IN

    # values of DG field on the nodes of the grid
    values::M

    # grid FE field exists on
    g::Grid
end

"""
    u = DGField(gfile, order, values)

Construct DG field from grid saved at `gfile` of order `order` with node values `values`.
"""
function DGField(order::Integer, values::AbstractMatrix, gfile::String)
    g = Grid(gfile, order)
    return DGField(order, values, g)
end
function DGField(values::AbstractMatrix, g::Grid)
    return DGField(g.order, values, g)
end
function DGField(value::Number, g::Grid)
    return DGField(value*ones(g.nt, g.nn), g)
end
function DGField(f::Function, g::Grid)
    return DGField([f(g.p[g.t[k, i], :]) for k=1:g.nt, i=1:g.nn], g)
end

# operations on DGFields
getindex(u::DGField, i, j) = u.values[i, j]

-(u::DGField, v::AbstractMatrix) = DGField(u.order, u.values - v, u.g)
-(u::DGField) = DGField(u.order, -u.values, u.g)
-(u::AbstractMatrix, v::DGField) = -(v - u)
-(u::DGField, v::DGField) = u - v.values

+(u::DGField, v::AbstractMatrix) = DGField(u.order, u.values + v, u.g)
+(u::AbstractMatrix, v::DGField) = v + u
+(u::DGField, v::DGField) = u + v.values

*(u::DGField, v::AbstractMatrix) = DGField(u.order, u.values .* v, u.g)
*(u::AbstractMatrix, v::DGField) = v * u
*(u::DGField, v::DGField) = u * v.values

/(u::DGField, v::AbstractMatrix) = DGField(u.order, u.values ./ v, u.g)
/(u::AbstractMatrix, v::DGField) = v / u
/(u::DGField, v::DGField) = u / v.values

struct FVField{V<:AbstractVector} <: Field
    # values of FV field on the elements of the grid
    values::V

    # grid FE field exists on
    g::Grid
end

"""
    u = FVField(gfile, values)

Construct FV field from grid saved at `gfile` with element values `values`.
"""
function FVField(values::AbstractVector, gfile::String)
    g = Grid(gfile, 1)
    return FVField(values, g)
end
function FVField(value::Number, g::Grid)
    return FVField(value*ones(g.nt), g)
end
function FVField(f::Function, g::Grid)
    return FVField([f(sum(g.p[g.t[k, i], :] for i=1:g.nn)/g.nn) for k=1:g.nt], g)
end

# operations on FVFields
getindex(u::FVField, k) = u.values[k]

-(u::FVField, v::AbstractVector) = FVField(u.values - v, u.g)
-(u::FVField) = FVField(-u.values, u.g)
-(u::AbstractVector, v::FVField) = -(v - u)
-(u::FVField, v::FVField) = u - v.values

+(u::FVField, v::AbstractVector) = FVField(u.order, u.values + v, u.g)
+(u::AbstractVector, v::FVField) = v + u
+(u::FVField, v::FVField) = u + v.values

*(u::FVField, v::AbstractVector) = FVField(u.values .* v, u.g)
*(u::AbstractVector, v::FVField) = v * u
*(u::FVField, v::FVField) = u * v.values

/(u::FVField, v::AbstractVector) = FVField(u.values ./ v, u.g)
/(u::AbstractVector, v::FVField) = v / u
/(u::FVField, v::FVField) = u / v.values

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
    ξ = transform_to_ref_el(x, u.g.p[u.g.t[k, 1:u.g.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[u.g.t[k, i]]*φ(u.g.sf, i, ξ) for i=1:u.g.nn)
end
function (u::DGField)(x, k)
    # transform to reference element
    ξ = transform_to_ref_el(x, u.g.p[u.g.t[k, 1:u.g.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[k, i]*φ(u.g.sf, i, ξ) for i=1:u.g.nn)
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
    ξ = transform_to_ref_el(x, u.g.p[u.g.t[k, 1:u.g.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[u.g.t[k, i]]*∂φ(u.g.sf, i, l, ξ)*u.g.J.Js[k, l, j] for i=1:u.g.nn, l=1:u.g.dim)
end
function ∂(u::DGField, x, k, j)
    # transform to reference element
    ξ = transform_to_ref_el(x, u.g.p[u.g.t[k, 1:u.g.dim+1], :])

    # sum weighted combinations of element k's basis functions at x
    return sum(u[k, i]*∂φ(u.g.sf, i, l, ξ)*u.g.J.Js[k, l, j] for i=1:u.g.nn, l=1:u.g.dim)
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