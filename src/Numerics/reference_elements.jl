"""
    p = reference_element_nodes(order, dim)

The nodes of a reference element of order `order` in `dim` dimensions.
"""
function reference_element_nodes(order, dim)
    if dim == 1
        if order == 0
            return [0.0]
        elseif order == 1
            return [-1.0
                     1.0]
        elseif order == 2
            return [-1.0
                     1.0
                     0.0]
        elseif order == 3
            return [-1.0
                     1.0
                    -1/3
                     1/3]
        else
            error("Unsupported reference element order `$order` for dimension `$dim`.")
        end
    elseif dim == 2
        if order == 0
            return 1/3*[1.0  1.0]
        elseif order == 1
            return [0.0  0.0
                    1.0  0.0
                    0.0  1.0]
        elseif order == 2
            return [0.0  0.0
                    1.0  0.0
                    0.0  1.0
                    0.5  0.0
                    0.5  0.5
                    0.0  0.5]
        elseif order == 3
            return [0.0  0.0
                    1.0  0.0
                    0.0  1.0
                    1/3  0.0
                    2/3  0.0
                    2/3  1/3
                    1/3  2/3
                    0.0  2/3
                    0.0  1/3
                    1/3  1/3]
        else
            error("Unsupported reference element order `$order` for dimension `$dim`.")
        end
    elseif dim == 3
        if order == 0
            return 1/3*[1.0  1.0  1.0]
        elseif order == 1
            return [0.0  0.0  0.0
                    1.0  0.0  0.0
                    0.0  1.0  0.0
                    0.0  0.0  1.0]
        elseif order == 2
            return [0.0  0.0  0.0
                    1.0  0.0  0.0
                    0.0  1.0  0.0
                    0.0  0.0  1.0
                    0.5  0.0  0.0
                    0.5  0.5  0.0
                    0.0  0.5  0.0
                    0.0  0.0  0.5
                    0.5  0.0  0.5
                    0.0  0.5  0.5]
        else
            error("Unsupported reference element order `$order` for dimension `$dim`.")
        end
    else
        error("Unsupported reference element dimension `$dim`.")
    end
end

"""
    x = transform_from_ref_el(ξ, p)

Transform point `ξ` defined on reference element to x defined on global element
with vertices `p`.
"""
function transform_from_ref_el(ξ, p)
    # dimension of space
    dim = size(p, 2)

    # build A
    A = zeros(dim, dim)
    for i=1:dim
        A[:, i] = p[i+1, :] - p[1, :]
    end

    # x = x₁ + A*ξ
    return p[1, :] + A*reshape(ξ, (dim, 1))
end

"""
    ξ = transform_to_ref_el(x, p)

Transform point `x` defined on global element with vertices `p` to reference element.
"""
function transform_to_ref_el(x, p)
    # dimension of space
    dim = size(p, 2)

    # build A
    A = zeros(dim, dim)
    for i=1:dim
        A[:, i] = p[i+1, :] - p[1, :]
    end

    # A*ξ = x - x₁
    return A\(x - p[1, :])
end