struct Lagrange
    order::Integer
end

function φ(::Line, sf::Lagrange, ξ, i)
    if sf.order == 1
        if i == 1
            return (1 - ξ)/2
        elseif i == 2
            return (1 + ξ)/2
        else
            throw(ArgumentError("Invalid shape function index: $i"))
        end
    elseif sf.order == 2
        if i == 1
            return (-ξ + ξ^2)/2
        elseif i == 2
            return (ξ + ξ^2)/2
        elseif i == 3
            return 1 - ξ^2
        else
            throw(ArgumentError("Invalid shape function index: $i"))
        end
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end
function ∂φ∂ξ(::Line, sf::Lagrange, ξ, i)
    if sf.order == 1
        if i == 1
            return -1/2
        elseif i == 2
            return 1/2
        else
            throw(ArgumentError("Invalid shape function index: $i"))
        end
    elseif sf.order == 2
        if i == 1
            return -1/2 + ξ
        elseif i == 2
            return 1/2 + ξ
        elseif i == 3
            return -2*ξ
        else
            throw(ArgumentError("Invalid shape function index: $i"))
        end
    else
        throw(ArgumentError("Unsupported order: $sf.order"))
    end
end

# #### 2D: Triangles ####

# struct Triangle{I<:Integer, M<:AbstractMatrix, V<:AbstractVector} <: AbstractElement
#     order::I
#     dim::I
#     n::I
#     p::M
#     C::M
#     quad_wts::V
#     quad_pts::M
# end

# function Triangle(; order)
#     if order == 1
#         p = [0.0  0.0 
#              1.0  0.0 
#              0.0  1.0]
#         C = [1.0  0.0  0.0
#             -1.0  1.0  0.0
#             -1.0  0.0  1.0]
#     elseif order == 2
#         p = [0.0  0.0  
#              1.0  0.0  
#              0.0  1.0  
#              0.5  0.0 
#              0.5  0.5
#              0.0  0.5]
#         C = [1.0  0.0  0.0  0.0  0.0  0.0
#             -3.0 -1.0  0.0  4.0  0.0  0.0
#             -3.0  0.0 -1.0  0.0  0.0  4.0
#              2.0  2.0  0.0 -4.0  0.0  0.0
#              4.0  0.0  0.0 -4.0  4.0 -4.0
#              2.0  0.0  2.0  0.0  0.0 -4.0]
#     else
#         error("Unsupported order")
#     end

#     n = size(p, 1)

#     # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html,
#     # exact up to degree 7 poly
#     quad_wts = [0.0235683681879057
#                 0.0353880678962995
#                 0.0225840492809381
#                 0.0054232259018275
#                 0.0441850885120943
#                 0.0663442161037005
#                 0.0423397245190619
#                 0.0101672595481725
#                 0.0441850885120943
#                 0.0663442161037005
#                 0.0423397245190619
#                 0.0101672595481725
#                 0.0235683681879057
#                 0.0353880678962995
#                 0.0225840492809381
#                 0.0054232259018275]
#     quad_pts = [0.0571041961  0.06546699455602246
#                 0.2768430136  0.05021012321401679
#                 0.5835904324  0.02891208422223085
#                 0.8602401357  0.009703785123906346
#                 0.0571041961  0.3111645522491480    
#                 0.2768430136  0.2386486597440242    
#                 0.5835904324  0.1374191041243166   
#                 0.8602401357  0.04612207989200404
#                 0.0571041961  0.6317312516508520   
#                 0.2768430136  0.4845083266559759    
#                 0.5835904324  0.2789904634756834    
#                 0.8602401357  0.09363778440799593
#                 0.0571041961  0.8774288093439775    
#                 0.2768430136  0.6729468631859832    
#                 0.5835904324  0.3874974833777692    
#                 0.8602401357  0.1300560791760936]  

#     return Triangle(order, 2, n, p, C, quad_wts, quad_pts)
# end

# function φ(el::Triangle, ξ, i)
#     if el.order == 1
#         c1 = el.C[1, i]
#         c2 = el.C[2, i]
#         c3 = el.C[3, i]
#         return c1 + ξ[1]*c2 + ξ[2]*c3
#     else el.order == 2
#         c1 = el.C[1, i]
#         c2 = el.C[2, i]
#         c3 = el.C[3, i]
#         c4 = el.C[4, i]
#         c5 = el.C[5, i]
#         c6 = el.C[6, i]
#         return c1 + c2*ξ[1] + c3*ξ[2] + c4*ξ[1]^2 + c5*ξ[1]*ξ[2] + c6*ξ[2]^2
#     end
# end
# function ∂φ(el::Triangle, ξ, i, j)
#     if j == 1
#         # ∂ξ
#         if el.order == 1
#             return el.C[2, i]
#         else el.order == 2
#             c2 = el.C[2, i]
#             c4 = el.C[4, i]
#             c5 = el.C[5, i]
#             return c2 + 2*c4*ξ[1] + c5*ξ[2]
#         end
#     elseif j == 2
#         # ∂η
#         if el.order == 1
#             return el.C[3, i]
#         else el.order == 2
#             c3 = el.C[3, i]
#             c5 = el.C[5, i]
#             c6 = el.C[6, i]
#             return c3 + c5*ξ[1] + 2*c6*ξ[2]
#         end
#     else
#         error("Invalid dimension of differentiation: `$j`.")
#     end
# end

# #### 3D: Wedges ####

# struct Wedge{I<:Integer, M<:AbstractMatrix, V<:AbstractVector} <: AbstractElement
#     order::I
#     dim::I
#     n::I
#     p::M
#     C::M
#     quad_wts::V
#     quad_pts::M
# end

# function Wedge(; order)
#     if order == 1
#         p = [0.0  0.0  0.0
#              1.0  0.0  0.0
#              0.0  1.0  0.0
#              0.0  0.0  1.0
#              1.0  0.0  1.0
#              0.0  1.0  1.0]
#         C = [1.0  0.0  0.0  0.0  0.0  0.0
#             -1.0  1.0  0.0  0.0  0.0  0.0
#             -1.0  0.0  1.0  0.0  0.0  0.0
#             -1.0  0.0  0.0  1.0  0.0  0.0
#              1.0 -1.0  0.0 -1.0  1.0  0.0
#              1.0  0.0 -1.0 -1.0  0.0  1.0]
#     elseif order == 2
#         p = [0.0  0.0  0.0
#              1.0  0.0  0.0
#              0.0  1.0  0.0
#              0.0  0.0  1.0
#              1.0  0.0  1.0
#              0.0  1.0  1.0
#              0.5  0.0  0.0
#              0.5  0.5  0.0
#              0.0  0.5  0.0
#              0.5  0.0  1.0
#              0.5  0.5  1.0
#              0.0  0.5  1.0]
#         C = [1.0  0.0  0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
#             -3.0 -1.0  0.0   0.0  0.0  0.0  4.0  0.0  0.0  0.0  0.0  0.0
#             -3.0  0.0 -1.0   0.0  0.0  0.0  0.0  0.0  4.0  0.0  0.0  0.0
#             -1.0  0.0  0.0   1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
#              3.0  1.0  0.0  -3.0 -1.0  0.0 -4.0  0.0  0.0  4.0  0.0  0.0
#              4.0  0.0  0.0   0.0  0.0  0.0 -4.0  4.0 -4.0  0.0  0.0  0.0
#              3.0  0.0  1.0  -3.0  0.0 -1.0  0.0  0.0 -4.0  0.0  0.0  4.0
#             -4.0  0.0  0.0   4.0  0.0  0.0  4.0 -4.0  4.0 -4.0  4.0 -4.0
#              2.0  2.0  0.0   0.0  0.0  0.0 -4.0  0.0  0.0  0.0  0.0  0.0
#              2.0  0.0  2.0   0.0  0.0  0.0  0.0  0.0 -4.0  0.0  0.0  0.0
#             -2.0 -2.0  0.0   2.0  2.0  0.0  4.0  0.0  0.0 -4.0  0.0  0.0
#             -2.0  0.0 -2.0   2.0  0.0  2.0  0.0  0.0  4.0  0.0  0.0 -4.0]
#     else
#         error("Unsupported order")
#     end

#     n = size(p, 1)

#     # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_wedge/quadrature_rules_wedge.html
#     # exact up to degree 4 poly
#     quad_wts = [0.0310252207886127
#                 0.0310252207886127
#                 0.0310252207886127
#                 0.0152710755076836
#                 0.0152710755076836
#                 0.0152710755076836
#                 0.0496403532617803
#                 0.0496403532617803
#                 0.0496403532617803
#                 0.0244337208122937
#                 0.0244337208122937
#                 0.0244337208122937
#                 0.0310252207886127
#                 0.0310252207886127
#                 0.0310252207886127
#                 0.0152710755076836
#                 0.0152710755076836
#                 0.0152710755076836]
#     quad_pts = [0.1081030181680702  0.4459484909159649  0.1127016653792583
#                 0.4459484909159649  0.1081030181680702  0.1127016653792583
#                 0.4459484909159649  0.4459484909159649  0.1127016653792583
#                 0.8168475729804585  0.0915762135097707  0.1127016653792583
#                 0.0915762135097707  0.8168475729804585  0.1127016653792583
#                 0.0915762135097707  0.0915762135097707  0.1127016653792583
#                 0.1081030181680702  0.4459484909159649  0.5000000000000000
#                 0.4459484909159649  0.1081030181680702  0.5000000000000000
#                 0.4459484909159649  0.4459484909159649  0.5000000000000000
#                 0.8168475729804585  0.0915762135097707  0.5000000000000000
#                 0.0915762135097707  0.8168475729804585  0.5000000000000000
#                 0.0915762135097707  0.0915762135097707  0.5000000000000000
#                 0.1081030181680702  0.4459484909159649  0.8872983346207417
#                 0.4459484909159649  0.1081030181680702  0.8872983346207417
#                 0.4459484909159649  0.4459484909159649  0.8872983346207417
#                 0.8168475729804585  0.0915762135097707  0.8872983346207417
#                 0.0915762135097707  0.8168475729804585  0.8872983346207417
#                 0.0915762135097707  0.0915762135097707  0.8872983346207417]

#     return Wedge(order, 3, n, p, C, quad_wts, quad_pts)
# end

# function φ(el::Wedge, ξ, i)
#     if el.order == 1
#         c1 = el.C[1, i]
#         c2 = el.C[2, i]
#         c3 = el.C[3, i]
#         c4 = el.C[4, i]
#         c5 = el.C[5, i]
#         c6 = el.C[6, i]
#         return c1 + c2*ξ[1] + c3*ξ[2] + c4*ξ[3] + c5*ξ[1]*ξ[3] + c6*ξ[2]*ξ[3]
#     else el.order == 2
#         c1 = el.C[1, i]
#         c2 = el.C[2, i]
#         c3 = el.C[3, i]
#         c4 = el.C[4, i]
#         c5 = el.C[5, i]
#         c6 = el.C[6, i]
#         c7 = el.C[7, i]
#         c8 = el.C[8, i]
#         c9 = el.C[9, i]
#         c10 = el.C[10, i]
#         c11 = el.C[11, i]
#         c12 = el.C[12, i]
#         return c1 + c2*ξ[1] + c3*ξ[2] + c4*ξ[3] + c5*ξ[1]*ξ[3] + c6*ξ[1]*ξ[2] + c7*ξ[2]*ξ[3] + c8*ξ[1]*ξ[2]*ξ[3] + 
#                c9*ξ[1]^2 + c10*ξ[2]^2 + c11*ξ[1]^2*ξ[3] + c12*ξ[2]^2*ξ[3]
#     end
# end
# function ∂φ(el::Wedge, ξ, i, j)
#     if j == 1
#         # ∂ξ
#         if el.order == 1
#             c2 = el.C[2, i]
#             c5 = el.C[5, i]
#             return c2 + c5*ξ[3]
#         else el.order == 2
#             c2 = el.C[2, i]
#             c5 = el.C[5, i]
#             c6 = el.C[6, i]
#             c8 = el.C[8, i]
#             c9 = el.C[9, i]
#             c11 = el.C[11, i]
#             return c2 + c5*ξ[3] + c6*ξ[2] + c8*ξ[2]*ξ[3] + c9*2*ξ[1] + c11*2*ξ[1]*ξ[3]
#         end
#     elseif j == 2
#         # ∂η
#         if el.order == 1
#             c3 = el.C[3, i]
#             c6 = el.C[6, i]
#             return c3 + c6*ξ[3]
#         else el.order == 2
#             c3 = el.C[3, i]
#             c6 = el.C[6, i]
#             c7 = el.C[7, i]
#             c8 = el.C[8, i]
#             c10 = el.C[10, i]
#             c12 = el.C[12, i]
#             return c3 + c6*ξ[1] + c7*ξ[3] + c8*ξ[1]*ξ[3] + c10*2*ξ[2] + c12*2*ξ[2]*ξ[3]
#         end
#     elseif j == 3
#         # ∂ζ
#         if el.order == 1
#             c4 = el.C[4, i]
#             c5 = el.C[5, i]
#             c6 = el.C[6, i]
#             return c4 + c5*ξ[1] + c6*ξ[2]
#         else el.order == 2
#             c4 = el.C[4, i]
#             c5 = el.C[5, i]
#             c7 = el.C[7, i]
#             c8 = el.C[8, i]
#             c11 = el.C[11, i]
#             c12 = el.C[12, i]
#             return c4 + c5*ξ[1] + c7*ξ[2] + c8*ξ[1]*ξ[2] + c11*ξ[1]^2 + c12*ξ[2]^2
#         end
#     else
#         error("Invalid dimension of differentiation: `$j`.")
#     end
# end

# #### Shortcuts ####

# φξ(el, ξ, i) = ∂φ(el, ξ, i, 1)
# φη(el, ξ, i) = ∂φ(el, ξ, i, 2)
# φζ(el, ξ, i) = ∂φ(el, ξ, i, 3)

# #### Transformations #### 

# """
#     A = transformation_matrix(el, p)

# Returns matrix `A` for the transformation ξ ↦ x, which is of the form 
#     x = A*ξ + b
# where
#     A for Line = (x₂ - x₁)/2,
#     A for Triangle = [x₂-x₁  x₃-x₁
#                       y₂-y₁  y₃-y₁],
#     A for Wedge = [x₂-x₁  x₃-x₁  0
#                    y₂-y₁  y₃-y₁  0
#                    0      0      z₄-z₁].
# """
# function transformation_matrix(el::Line, p)
#     return (p[2] - p[1])/2
# end
# function transformation_matrix(el::Triangle, p)
#     return [p[j+1, i] - p[1, i] for i=1:2, j=1:2]
# end
# function transformation_matrix(el::Wedge, p)
#     return [max(i, j) ≤ 2 || i == j ? p[j+1, i] - p[1, i] : 0. for i=1:3, j=1:3]
# end

# """
#     b = transformation_vector(el, p)

# Returns vector `b` for the transformation ξ ↦ x, which is of the form 
#     x = A*ξ + b
# where
#     b for Line = (x₂ + x₁)/2,
#     b for Triangle = [x₁, y₁],
#     b for Wedge = [x₁, y₁, z₁].
# """
# function transformation_vector(el::Line, p)
#     return (p[1] + p[2])/2
# end
# function transformation_vector(el::Union{Triangle, Wedge}, p)
#     return p[1, :]
# end

# """
#     ξ = transform_to_ref_el(el, x, p)

# Returns coordinates `ξ` in reference element that map to `x` in the element
# defined by the nodes `p`.
# """
# function transform_to_ref_el(el, x, p)
#     A = transformation_matrix(el, p)
#     b = transformation_vector(el, p)
#     return A\(x .- b)
# end

# """
#     x = transform_from_ref_el(el, ξ, p)

# Returns coordinates `x` in element defined by the nodes `p` that map to `ξ` 
# in the reference element.
# """
# function transform_from_ref_el(el, ξ, p)
#     A = transformation_matrix(el, p)
#     b = transformation_vector(el, p)
#     return A*ξ .+ b
# end

# #### Quadrature ####

# φ_quad_pts(el::AbstractElement) = [φ(el, el.quad_pts[i_quad, :], i) for i ∈ 1:el.n, i_quad ∈ eachindex(el.quad_wts)]
# ∂φ_quad_pts(el::AbstractElement) = [∂φ(el, el.quad_pts[i_quad, :], i, j) for i ∈ 1:el.n, j ∈ 1:el.dim, i_quad ∈ eachindex(el.quad_wts)]

# #### Some useful finite element matrices ####

# stiffness_matrix(el::AbstractElement) = [ref_el_quad(ξ->∂φ(el, ξ, i, k)*∂φ(el, ξ, j, l), el) for k=1:el.dim, l=1:el.dim, i=1:el.n, j=1:el.n]
# mass_matrix(el::AbstractElement) = [ref_el_quad(ξ->φ(el, ξ, i)*φ(el, ξ, j), el) for i=1:el.n, j=1:el.n]