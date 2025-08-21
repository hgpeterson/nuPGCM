abstract type AbstractElement end

# struct Point{T} <: AbstractElement 
#     x₁::T
#     x₂::T
#     x₃::T
# end

# function Point(x₁) 
#     return Point(x₁, zero(x₁), zero(x₁))
# end
# function Point(x₁, x₂) 
#     x₁, x₂ = promote(x₁, x₂)
#     return Point(x₁, x₂, zero(x₁))
# end
# function Point(x₁, x₂, x₃) 
#     x₁, x₂, x₃ = promote(x₁, x₂, x₃)
#     return Point(x₁, x₂, x₃)
# end

struct Point <: AbstractElement end
struct Line <: AbstractElement end
struct Triangle <: AbstractElement end
struct Tetrahedron <: AbstractElement end

dimension(::Point) = 0
dimension(::Line) = 1
dimension(::Triangle) = 2
dimension(::Tetrahedron) = 3

reference_element(::Line) = [-1.0, 
                              1.0]
reference_element(::Triangle) = [0.0  0.0 
                                 1.0  0.0
                                 0.0  1.0]
reference_element(::Tetrahedron) = [0.0  0.0  0.0
                                    1.0  0.0  0.0
                                    0.0  1.0  0.0
                                    0.0  0.0  1.0]

function get_element_type(dim)
    if dim == 0
        return Point()
    elseif dim == 1
        return Line()
    elseif dim == 2
        return Triangle()
    elseif dim == 3
        return Tetrahedron()
    else
        throw(ArgumentError("Invalid dimension: $dim"))
    end
end

"""
    A, b = transformation_matrix_vector(element, vertices)

Compute matrix/vector to transform from reference element to element defined by `vertices`.

Returns matrix ``A`` and vector ``b`` such that the transformation is given by
```math
    x = x₁ + ∂x/∂ξ (ξ - ξ₁)
````
where ``ξ`` is the coordinate in the reference element.

`vertices` must be an ``n × 3`` matrix with each row corresponding to the ``(x, y, z)``
coordinates of one of the ``n`` vertices of the element.
"""
function build_jacobian(::Line, vertices)
    return norm(vertices[2, :] - vertices[1, :])
end

function build_jacobian(el::Triangle, vertices)
    v₁, v₂ = get_local_basis(el, vertices)
    e₁, e₂ = get_local_basis_perp(v₁, v₂)
    return [dot(v₁, e₁)  dot(v₂, e₁)
            dot(v₁, e₂)  dot(v₂, e₂)]
end
function get_local_basis(::Triangle, vertices)
    x₁ = vertices[1, :]
    x₂ = vertices[2, :]
    x₃ = vertices[3, :]
    v₁ = x₂ - x₁
    v₂ = x₃ - x₁
    return v₁, v₂
end
function get_local_basis_perp(el::Triangle, vertices)
    v₁, v₂ = get_local_basis(el, vertices)
    return get_local_basis_perp(v₁, v₂)
end
function get_local_basis_perp(v₁, v₂)
    e₁ = v₁/norm(v₁)
    e₂ = cross(cross(v₁, v₂), v₁)
    e₂ /= norm(e₂)
    return e₁, e₂
end

function build_jacobian(::Tetrahedron, vertices)
    return [vertices[j+1, i] - vertices[1, i] for i in 1:3, j in 1:3]
end

"""
    ξ = transform_to_reference(el::AbstractElement, x, vertices)

Transform the point ``x`` in the element defined by `vertices` to ``ξ`` in the reference element.
"""
function transform_to_reference(el::Line, x, vertices)
    ∂x∂ξ = build_jacobian(el, vertices)
    x₁ = vertices[1, :]
    ξ₁ = reference_element(el)[1]
    return ξ₁ + (x .- x₁)/∂x∂ξ
end
function transform_to_reference(el::Triangle, x, vertices)
    ∂x∂ξ = build_jacobian(el, vertices)
    e₁, e₂ = get_local_basis_perp(el, vertices)
    ∂x = x - x₁
    ∂x_e = [dot(∂x, e₁)  dot(∂x, e₂)]
    ξ₁ = reference_element(el)[1, :]
    return ξ₁ .+ ∂x∂ξ \ ∂x_e
end
function transform_to_reference(el::Tetrahedron, x, vertices)
    ∂x∂ξ = build_jacobian(el, vertices)
    x₁ = vertices[1, :]
    ξ₁ = reference_element(el)[1, :]
    return ξ₁ .+ ∂x∂ξ\(x .- x₁)
end

# """
#     x = transform_from_reference(el::AbstractElement, ξ, vertices)

# Transform the point ``ξ`` in the reference element to ``x`` in the element defined by `vertices`.
# """
# function transform_from_reference(el::AbstractElement, ξ, vertices)
#     ∂x∂ξ = build_jacobian(el, vertices)
#     x₁ = vertices[1, :]
#     ξ₁ = reference_element(el)[1, :]
#     return x₁ .+ ∂x∂ξ*(ξ .- ξ₁)
# end