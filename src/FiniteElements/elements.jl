abstract type AbstractElement end

struct Line <: AbstractElement end
struct Triangle <: AbstractElement end
struct Tetrahedron <: AbstractElement end

reference_element(::Line) = [-1.0, 
                              1.0]
reference_element(::Triangle) = [0.0  0.0 
                                 1.0  0.0
                                 0.0  1.0]
reference_element(::Tetrahedron) = [0.0  0.0  0.0
                                    1.0  0.0  0.0
                                    0.0  1.0  0.0
                                    0.0  0.0  1.0]

"""
    A, b = transformation_matrix_vector(element, vertices)

Compute matrix/vector to transform from reference element to element defined by `vertices`.

Returns matrix ``A`` and vector ``b`` such that the transformation is given by
```math
    x = A ξ + b
````
where ``ξ`` is the coordinate in the reference element.

`vertices` must be an ``n × 3`` matrix with each row corresponding to the ``(x, y, z)``
coordinates of one of the ``n`` vertices of the element.
"""
function transformation_matrix_vector(::Line, vertices)
    A = (vertices[2, :] - vertices[1, :])/2
    b = (vertices[1, :] + vertices[2, :])/2
    return A, b
end
function transformation_matrix_vector(::Triangle, vertices)
    p1 = vertices[1, :]
    p2 = vertices[2, :]
    p3 = vertices[3, :]
    v1 = p2 - p1
    v2 = p3 - p1
    e_x′ = v1/norm(v1)
    e_y′ = cross(cross(v1, v2), v1)
    e_y′ /= norm(e_y′)
    A = [dot(v1, e_x′)  dot(v2, e_x′)
         dot(v1, e_y′)  dot(v2, e_y′)]
    b = [0., 0.]
    return A, b
end
function transformation_matrix_vector(::Tetrahedron, vertices)
    A = [vertices[j+1, i] - vertices[1, i] for i in 1:3, j in 1:3]
    b = vertices[1, :]
    return A, b
end

"""
    ξ = transform_to_reference(e::AbstractElement, x, vertices)

Transform the point ``x`` in the element defined by `vertices` to ``ξ`` in the reference element.
"""
function transform_to_reference(e::AbstractElement, x, vertices)
    A, b = transformation_matrix(e, vertices)
    return A\(x .- b)
end

"""
    x = transform_from_reference(e::AbstractElement, ξ, vertices)

Transform the point ``ξ`` in the reference element to ``x`` in the element defined by `vertices`.
"""
function transform_from_reference(e::AbstractElement, ξ, vertices)
    A, b = transformation_matrix(e, vertices)
    return A*ξ .+ b
end