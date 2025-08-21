abstract type AbstractShapeFunction end

struct Lagrange <: AbstractShapeFunction
    order::Integer
end

function φ(::Line, sf::Lagrange, ξ)
    if sf.order == 1
        return [(1 - ξ)/2
                (1 + ξ)/2]
    elseif sf.order == 2
        return [(-ξ + ξ^2)/2
                ( ξ + ξ^2)/2
                  1 - ξ^2]
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end
function ∂φ∂ξ(::Line, sf::Lagrange, ξ)
    if sf.order == 1
        return [-1/2
                 1/2]
    elseif sf.order == 2
        return [-1/2 + ξ
                 1/2 + ξ
                    -2*ξ]
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end
∇φ(::Line, sf::Lagrange, ξ) = [∂φ∂ξ(Line(), sf, ξ)]

#         C = [1.0  0.0  0.0
#             -1.0  1.0  0.0
#             -1.0  0.0  1.0]
#         C = [1.0  0.0  0.0  0.0  0.0  0.0
#             -3.0 -1.0  0.0  4.0  0.0  0.0
#             -3.0  0.0 -1.0  0.0  0.0  4.0
#              2.0  2.0  0.0 -4.0  0.0  0.0
#              4.0  0.0  0.0 -4.0  4.0 -4.0
#              2.0  0.0  2.0  0.0  0.0 -4.0]
function φ(::Triangle, sf::Lagrange, ξ)
    if sf.order == 1
        return [(1 - ξ[1] - ξ[2])
                ξ[1]
                ξ[2]]
    elseif sf.order == 2
        return [1 - 3*ξ[1] - 3*ξ[2] + 2*ξ[1]^2 + 4*ξ[1]*ξ[2] + 2*ξ[2]^2
                -ξ[1] + 2*ξ[1]^2
                -ξ[2] + 2*ξ[2]^2
                4*ξ[1] - 4*ξ[1]^2 - 4*ξ[1]*ξ[2]
                4*ξ[1]*ξ[2]
                4*ξ[2] - 4*ξ[1]*ξ[2] - 4*ξ[2]^2]
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end
function ∂φ∂ξ(::Triangle, sf::Lagrange, ξ)
    if sf.order == 1
        return [-1
                 1
                 0]
    elseif sf.order == 2
        return [-3 + 4*ξ[1] + 4*ξ[2]
                 4*ξ[1] - 1
                 0
                 4 - 8*ξ[1] - 4*ξ[2]
                 4*ξ[2]
                 -4*ξ[2]]
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end
function ∂φ∂η(::Triangle, sf::Lagrange, ξ)
    if sf.order == 1
        return [-1
                 0
                 1]
    elseif sf.order == 2
        return [-3 + 4*ξ[1] + 4*ξ[2]
                 0
                 4*ξ[2] - 1
                 -4*ξ[1]
                 4*ξ[1]
                 4 - 4*ξ[1] - 8*ξ[2]]
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end

function ∇φ(::Triangle, sf::Lagrange, ξ) 
    φξ = ∂φ∂ξ(Triangle(), sf, ξ)
    φη = ∂φ∂η(Triangle(), sf, ξ)
    return [[φξ[i], φη[i]] for i in eachindex(φξ)]
end