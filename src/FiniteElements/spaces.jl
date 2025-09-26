abstract type AbstractFESpace end

################################################################################

struct P1 <: AbstractFESpace end

n_dofs(::Line,        ::P1) = 2
n_dofs(::Triangle,    ::P1) = 3
n_dofs(::Tetrahedron, ::P1) = 4

φ(::Line, ::P1, ξ) = [(1 - ξ)/2, (1 + ξ)/2]
∂φ∂ξ(::Line, ::P1, ξ) = eltype(ξ).([-1/2, 1/2])

φ(::Triangle, ::P1, ξ) = [(1 - ξ[1] - ξ[2]), ξ[1], ξ[2]]
∂φ∂ξ(::Triangle, ::P1, ξ) = eltype(ξ).([-1, 1, 0])
∂φ∂η(::Triangle, ::P1, ξ) = eltype(ξ).([-1, 0, 1])

################################################################################

struct P2 <: AbstractFESpace end

n_dofs(::Line,        ::P2) = 3
n_dofs(::Triangle,    ::P2) = 6
n_dofs(::Tetrahedron, ::P2) = 10

φ(::Line, ::P2, ξ) = [(-ξ + ξ^2)/2, (ξ + ξ^2)/2, 1 - ξ^2]
∂φ∂ξ(::Line, ::P2, ξ) = [-1/2 + ξ, 1/2 + ξ, -2ξ]

φ(::Triangle, ::P2, ξ) = [1 - 3ξ[1] - 3ξ[2] + 2ξ[1]^2 + 4ξ[1]*ξ[2] + 2ξ[2]^2,
                         -ξ[1] + 2ξ[1]^2,
                         -ξ[2] + 2ξ[2]^2,
                          4ξ[1] - 4ξ[1]^2 - 4ξ[1]*ξ[2],
                          4ξ[1]*ξ[2],
                          4ξ[2] - 4*ξ[1]*ξ[2] - 4ξ[2]^2]
∂φ∂ξ(::Triangle, ::P2, ξ) = [-3 + 4ξ[1] + 4ξ[2]
                              4ξ[1] - 1
                              0
                              4 - 8ξ[1] - 4ξ[2]
                              4ξ[2]
                             -4ξ[2]]
∂φ∂η(::Triangle, ::P2, ξ) = [-3 + 4ξ[1] + 4ξ[2]
                              0
                              4ξ[2] - 1
                             -4ξ[1]
                              4ξ[1]
                              4 - 4ξ[1] - 8ξ[2]]

################################################################################

struct Bubble <: AbstractFESpace end

n_dofs(::AbstractElement, ::Bubble) = 1

φ(::Line, ::Bubble, ξ) = (1 + ξ)*(1 - ξ)
∂φ∂ξ(::Line, ::Bubble, ξ) = -2ξ

φ(::Triangle, ::Bubble, ξ) = 27ξ[1]*ξ[2]*(1 - ξ[1] - ξ[2])
∂φ∂ξ(::Triangle, ::Bubble, ξ) = 27ξ[2]*(1 - 2ξ[1] - ξ[2])
∂φ∂η(::Triangle, ::Bubble, ξ) = 27ξ[1]*(1 - ξ[1] - 2ξ[2])

φ(::Tetrahedron, ::Bubble, ξ) = 256ξ[1]*ξ[2]*ξ[3]*(1 - ξ[1] - ξ[2] - ξ[3])
∂φ∂ξ(::Tetrahedron, ::Bubble, ξ) = 256ξ[2]*ξ[3]*(1 - 2ξ[1] - ξ[2] - ξ[3])
∂φ∂η(::Tetrahedron, ::Bubble, ξ) = 256ξ[1]*ξ[3]*(1 - ξ[1] - 2ξ[2] - ξ[3])
∂φ∂ζ(::Tetrahedron, ::Bubble, ξ) = 256ξ[1]*ξ[2]*(1 - ξ[1] - ξ[2] - 2ξ[3]) 

################################################################################

struct Mini <: AbstractFESpace end

n_dofs(el::AbstractElement, ::Mini) = n_dofs(el, P1()) + n_dofs(el, Bubble())

φ(el::AbstractElement, ::Mini, ξ) = vcat(φ(el, P1(), ξ), φ(el, Bubble(), ξ))
∂φ∂ξ(el::AbstractElement, ::Mini, ξ) = vcat(∂φ∂ξ(el, P1(), ξ), ∂φ∂ξ(el, Bubble(), ξ))
∂φ∂η(el::AbstractElement, ::Mini, ξ) = vcat(∂φ∂η(el, P1(), ξ), ∂φ∂η(el, Bubble(), ξ))
∂φ∂ζ(el::AbstractElement, ::Mini, ξ) = vcat(∂φ∂ζ(el, P1(), ξ), ∂φ∂ζ(el, Bubble(), ξ))

################################################################################

∇φ(el::Line, space::AbstractFESpace, ξ) = [∂φ∂ξ(el, space, ξ)]

function ∇φ(el::Triangle, space::AbstractFESpace, ξ) 
    φξ = ∂φ∂ξ(el, space, ξ)
    φη = ∂φ∂η(el, space, ξ)
    return [[φξ[i], φη[i]] for i in eachindex(φξ)]
end

function ∇φ(el::Tetrahedron, space::AbstractFESpace, ξ) 
    φξ = ∂φ∂ξ(el, space, ξ)
    φη = ∂φ∂η(el, space, ξ)
    φζ = ∂φ∂ζ(el, space, ξ)
    return [[φξ[i], φη[i], φζ[i]] for i in eachindex(φξ)]
end