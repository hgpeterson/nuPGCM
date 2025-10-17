abstract type AbstractFESpace end

################################################################################

struct ShapeFunction{E <: AbstractElement, S <: AbstractFESpace}
    element::E
    space::S
end

# make ShapeFunction instances callable: φ(ξ, i) -> value of the i-th shape function at ξ
function (φ::ShapeFunction)(ξ, i)
    return shape_func_value(φ.element, φ.space, ξ, i)
end

# ∇(φ) returns a callable ∇φ(ξ, i) that gives the gradient (in reference coords) of the i-th shape function
function ∇(φ::ShapeFunction)
    return (ξ, i) -> begin
        return shape_func_gradient(φ.element, φ.space, ξ, i)
    end
end

n_dofs(φ::ShapeFunction) = n_dofs(φ.element, φ.space)

################################################################################

struct P1 <: AbstractFESpace end

n_dofs(::Line,        ::P1) = 2
n_dofs(::Triangle,    ::P1) = 3
n_dofs(::Tetrahedron, ::P1) = 4

shape_func_value(::Line, ::P1, ξ, ::Val{1}) = (1 - ξ)/2
shape_func_value(::Line, ::P1, ξ, ::Val{2}) = (1 + ξ)/2
shape_func_gradient(::Line, ::P1, ξ, ::Val{1}) = eltype(ξ).(-1/2) 
shape_func_gradient(::Line, ::P1, ξ, ::Val{2}) = eltype(ξ).(1/2) 

shape_func_value(::Triangle, ::P1, ξ, ::Val{1}) = 1 - ξ[1] - ξ[2]
shape_func_value(::Triangle, ::P1, ξ, ::Val{2}) = ξ[1]
shape_func_value(::Triangle, ::P1, ξ, ::Val{3}) = ξ[2]
shape_func_gradient(::Triangle, ::P1, ξ, ::Val{1}) = eltype(ξ).([-1, -1])
shape_func_gradient(::Triangle, ::P1, ξ, ::Val{2}) = eltype(ξ).([1, 0])
shape_func_gradient(::Triangle, ::P1, ξ, ::Val{3}) = eltype(ξ).([0, 1])

################################################################################

struct P2 <: AbstractFESpace end

n_dofs(::Line,        ::P2) = 3
n_dofs(::Triangle,    ::P2) = 6
n_dofs(::Tetrahedron, ::P2) = 10

shape_func_value(::Line, ::P2, ξ, ::Val{1}) = (-ξ + ξ^2)/2
shape_func_value(::Line, ::P2, ξ, ::Val{2}) = (ξ + ξ^2)/2
shape_func_value(::Line, ::P2, ξ, ::Val{3}) = 1 - ξ^2
shape_func_gradient(::Line, ::P2, ξ, ::Val{1}) = -1/2 + ξ
shape_func_gradient(::Line, ::P2, ξ, ::Val{2}) = 1/2 + ξ
shape_func_gradient(::Line, ::P2, ξ, ::Val{3}) = -2ξ

shape_func_value(::Triangle, ::P2, ξ, ::Val{1}) = 1 - 3ξ[1] - 3ξ[2] + 2ξ[1]^2 + 4ξ[1]*ξ[2] + 2ξ[2]^2
shape_func_value(::Triangle, ::P2, ξ, ::Val{2}) = -ξ[1] + 2ξ[1]^2
shape_func_value(::Triangle, ::P2, ξ, ::Val{3}) = -ξ[2] + 2ξ[2]^2
shape_func_value(::Triangle, ::P2, ξ, ::Val{4}) = 4ξ[1] - 4ξ[1]^2 - 4ξ[1]*ξ[2]
shape_func_value(::Triangle, ::P2, ξ, ::Val{5}) = 4ξ[1]*ξ[2]
shape_func_value(::Triangle, ::P2, ξ, ::Val{6}) = 4ξ[2] - 4*ξ[1]*ξ[2] - 4ξ[2]^2
shape_func_gradient(::Triangle, ::P2, ξ, ::Val{1}) = [-3 + 4ξ[1] + 4ξ[2], -3 + 4ξ[1] + 4ξ[2]]
shape_func_gradient(::Triangle, ::P2, ξ, ::Val{2}) = [4ξ[1] - 1, 0]
shape_func_gradient(::Triangle, ::P2, ξ, ::Val{3}) = [0, 4ξ[2] - 1]
shape_func_gradient(::Triangle, ::P2, ξ, ::Val{4}) = [4 - 8ξ[1] - 4ξ[2], -4ξ[1]]
shape_func_gradient(::Triangle, ::P2, ξ, ::Val{5}) = [4ξ[2], 4ξ[1]]
shape_func_gradient(::Triangle, ::P2, ξ, ::Val{6}) = [-4ξ[2], 4 - 4ξ[1] - 8ξ[2]]

################################################################################

struct Bubble <: AbstractFESpace end

n_dofs(::AbstractElement, ::Bubble) = 1

shape_func_value(::Line, ::Bubble, ξ) = (1 + ξ)*(1 - ξ)
shape_func_gradient(::Line, ::Bubble, ξ) = -2ξ

shape_func_value(::Triangle, ::Bubble, ξ) = 27ξ[1]*ξ[2]*(1 - ξ[1] - ξ[2])
shape_func_gradient(::Triangle, ::Bubble, ξ) = [27ξ[2]*(1 - 2ξ[1] - ξ[2]), 
                                                27ξ[1]*(1 - ξ[1] - 2ξ[2])]

shape_func_value(::Tetrahedron, ::Bubble, ξ) = 256ξ[1]*ξ[2]*ξ[3]*(1 - ξ[1] - ξ[2] - ξ[3])
shape_func_gradient(::Tetrahedron, ::Bubble, ξ) = [256ξ[2]*ξ[3]*(1 - 2ξ[1] - ξ[2] - ξ[3]),
                                                   256ξ[1]*ξ[3]*(1 - ξ[1] - 2ξ[2] - ξ[3]),
                                                   256ξ[1]*ξ[2]*(1 - ξ[1] - ξ[2] - 2ξ[3])]

################################################################################

struct Mini <: AbstractFESpace end

n_dofs(el::AbstractElement, ::Mini) = n_dofs(el, P1()) + n_dofs(el, Bubble())

shape_func_value(el::AbstractElement, ::Mini, ξ, ::Val{1}) = shape_func_value(el, P1(), ξ, val(1))
shape_func_value(el::AbstractElement, ::Mini, ξ, ::Val{2}) = shape_func_value(el, P1(), ξ, val(2))
shape_func_value(el::AbstractElement, ::Mini, ξ, ::Val{3}) = shape_func_value(el, P1(), ξ, val(3))
shape_func_value(el::AbstractElement, ::Mini, ξ, ::Val{4}) = shape_func_value(el, Bubble(), ξ)
shape_func_gradient(el::AbstractElement, ::Mini, ξ) = vcat(shape_func_gradient(el, P1(), ξ), 
                                                           shape_func_gradient(el, Bubble(), ξ))