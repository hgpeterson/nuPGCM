abstract type AbstractElement end

#### 1D: Lines ####

struct Line{I<:Integer, V<:AbstractVector, M<:AbstractMatrix} <: AbstractElement
    order::I
    n::I
    p_ref::V
    C::M
end

function Line(; order)
    if order == 1
        p_ref = [-1.0, 1.0]
        C = [0.5  0.5
            -0.5  0.5]
    elseif order == 2
        p_ref = [-1.0, 1.0, 0.0]
        C = [0.0  0.0  1.0  
            -0.5  0.5  0.0  
             0.5  0.5  1.0]
    else
        error("Unsupported order")
    end

    n = size(p_ref, 1)

    return Line(order, n, p_ref, C)
end

function φ(el::Line, ξ, i)
    if el.order == 1
        return ([1 ξ[1]]*el.C[:, i])[1]
    else el.order == 2
        return ([1 ξ[1] ξ[1]^2]*el.C[:, i])[1]
    end
end
function φξ(el::Line, ξ, i)
    if el.order == 1
        return ([0 1]*el.C[:, i])[1]
    else el.order == 2
        return ([0 1 2ξ[1]]*el.C[:, i])[1]
    end
end

#### 2D: Triangles ####

struct Triangle{I<:Integer, M<:AbstractMatrix} <: AbstractElement
    order::I
    n::I
    p_ref::M
    C::M
end

function Triangle(; order)
    if order == 1
        p_ref = [0.0  0.0  0.0 
                 1.0  0.0  0.0 
                 0.0  1.0  0.0]
        C = [1.0  0.0  0.0
            -1.0  1.0  0.0
            -1.0  0.0  1.0]
    elseif order == 2
        p_ref = [0.0  0.0  
                 1.0  0.0  
                 0.0  1.0  
                 0.5  0.0 
                 0.5  0.5
                 0.0  0.5]
        C = [1.0  0.0  0.0  0.0  0.0  0.0
            -3.0 -1.0  0.0  4.0  0.0  0.0
            -3.0  0.0 -1.0  0.0  0.0  4.0
             2.0  2.0  0.0 -4.0  0.0  0.0
             4.0  0.0  0.0 -4.0  4.0 -4.0
             2.0  0.0  2.0  0.0  0.0 -4.0]
    else
        error("Unsupported order")
    end

    n = size(p_ref, 1)

    return Triangle(order, n, p_ref, C)
end

function φ(el::Triangle, ξ, i)
    if el.order == 1
        return ([1 ξ[1] ξ[2]]*el.C[:, i])[1]
    else el.order == 2
        return ([1 ξ[1] ξ[2] ξ[1]^2 ξ[1]*ξ[2] ξ[2]^2]*el.C[:, i])[1]
    end
end
function φξ(el::Triangle, ξ, i)
    if el.order == 1
        return ([0 1 0]*el.C[:, i])[1]
    else el.order == 2
        return ([0 1 0 2ξ[1] ξ[2] 0]*el.C[:, i])[1]
    end
end
function φη(el::Triangle, ξ, i)
    if el.order == 1
        return ([0 0 1]*el.C[:, i])[1]
    else el.order == 2
        return ([0 0 1 0 ξ[1] 2ξ[2]]*el.C[:, i])[1]
    end
end

#### 3D: Wedges ####

struct Wedge{I<:Integer, M<:AbstractMatrix} <: AbstractElement
    order::I
    n::I
    p_ref::M
    C::M
end

function Wedge(; order)
    if order == 1
        p_ref = [0.0  0.0  0.0
                 1.0  0.0  0.0
                 0.0  1.0  0.0
                 0.0  0.0  1.0
                 1.0  0.0  1.0
                 0.0  1.0  1.0]
        C = [1.0  0.0  0.0  0.0  0.0  0.0
            -1.0  1.0  0.0  0.0  0.0  0.0
            -1.0  0.0  1.0  0.0  0.0  0.0
            -1.0  0.0  0.0  1.0  0.0  0.0
             1.0 -1.0  0.0 -1.0  1.0  0.0
             1.0  0.0 -1.0 -1.0  0.0  1.0]
    elseif order == 2
        p_ref = [0.0  0.0  0.0
                 1.0  0.0  0.0
                 0.0  1.0  0.0
                 0.0  0.0  1.0
                 1.0  0.0  1.0
                 0.0  1.0  1.0
                 0.5  0.0  0.0
                 0.5  0.5  0.0
                 0.0  0.5  0.0
                 0.5  0.0  1.0
                 0.5  0.5  1.0
                 0.0  0.5  1.0]
        C = [1.0  0.0  0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            -3.0 -1.0  0.0   0.0  0.0  0.0  4.0  0.0  0.0  0.0  0.0  0.0
            -3.0  0.0 -1.0   0.0  0.0  0.0  0.0  0.0  4.0  0.0  0.0  0.0
            -1.0  0.0  0.0   1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
             3.0  1.0  0.0  -3.0 -1.0  0.0 -4.0  0.0  0.0  4.0  0.0  0.0
             4.0  0.0  0.0   0.0  0.0  0.0 -4.0  4.0 -4.0  0.0  0.0  0.0
             3.0  0.0  1.0  -3.0  0.0 -1.0  0.0  0.0 -4.0  0.0  0.0  4.0
            -4.0  0.0  0.0   4.0  0.0  0.0  4.0 -4.0  4.0 -4.0  4.0 -4.0
             2.0  2.0  0.0   0.0  0.0  0.0 -4.0  0.0  0.0  0.0  0.0  0.0
             2.0  0.0  2.0   0.0  0.0  0.0  0.0  0.0 -4.0  0.0  0.0  0.0
            -2.0 -2.0  0.0   2.0  2.0  0.0  4.0  0.0  0.0 -4.0  0.0  0.0
            -2.0  0.0 -2.0   2.0  0.0  2.0  0.0  0.0  4.0  0.0  0.0 -4.0]
    else
        error("Unsupported order")
    end

    n = size(p_ref, 1)

    return Wedge(order, n, p_ref, C)
end

function φ(el::Wedge, ξ, i)
    if el.order == 1
        return ([1 ξ[1] ξ[2] ξ[3] ξ[1]*ξ[3] ξ[2]*ξ[3]]*el.C[:, i])[1]
    else el.order == 2
        return ([1 ξ[1] ξ[2] ξ[3] ξ[1]*ξ[3] ξ[1]*ξ[2] ξ[2]*ξ[3] ξ[1]*ξ[2]*ξ[3] ξ[1]^2 ξ[2]^2 ξ[1]^2*ξ[3] ξ[2]^2*ξ[3]]*el.C[:, i])[1]
    end
end
function φξ(el::Wedge, ξ, i)
    if el.order == 1
        return ([0 1 0 0 ξ[3] 0]*el.C[:, i])[1]
    else el.order == 2
        return ([0 1 0 0 ξ[3] ξ[2] 0 ξ[2]*ξ[3] 2*ξ[1] 0 2*ξ[1]*ξ[3] 0]*el.C[:, i])[1]
    end
end
function φη(el::Wedge, ξ, i)
    if el.order == 1
        return ([0 0 1 0 0 ξ[3]]*el.C[:, i])[1]
    else el.order == 2
        return ([0 0 1 0 0 ξ[1] ξ[3] ξ[1]*ξ[3] 0 2*ξ[2] 0 2*ξ[2]*ξ[3]]*el.C[:, i])[1]
    end
end
function φζ(el::Wedge, ξ, i)
    if el.order == 1
        return ([0 0 0 1 ξ[1] ξ[2]]*el.C[:, i])[1]
    else el.order == 2
        return ([0 0 0 1 ξ[1] 0 ξ[2] ξ[1]*ξ[2] 0 0 ξ[1]^2 ξ[2]^2]*el.C[:, i])[1]
    end
end

#### Jacobians ####

xξ(el::AbstractElement, ξ, p) = sum(φξ(el, ξ, i)*p[i, 1] for i ∈ axes(p, 1))
yξ(el::AbstractElement, ξ, p) = sum(φξ(el, ξ, i)*p[i, 2] for i ∈ axes(p, 1))
zξ(el::AbstractElement, ξ, p) = sum(φξ(el, ξ, i)*p[i, 3] for i ∈ axes(p, 1))
xη(el::AbstractElement, ξ, p) = sum(φη(el, ξ, i)*p[i, 1] for i ∈ axes(p, 1))
yη(el::AbstractElement, ξ, p) = sum(φη(el, ξ, i)*p[i, 2] for i ∈ axes(p, 1))
zη(el::AbstractElement, ξ, p) = sum(φη(el, ξ, i)*p[i, 3] for i ∈ axes(p, 1))
xζ(el::AbstractElement, ξ, p) = sum(φζ(el, ξ, i)*p[i, 1] for i ∈ axes(p, 1))
yζ(el::AbstractElement, ξ, p) = sum(φζ(el, ξ, i)*p[i, 2] for i ∈ axes(p, 1))
zζ(el::AbstractElement, ξ, p) = sum(φζ(el, ξ, i)*p[i, 3] for i ∈ axes(p, 1))
J(el::AbstractElement, ξ, p) = inv([xξ(el, ξ, p) xη(el, ξ, p) xζ(el, ξ, p)
                                    yξ(el, ξ, p) yη(el, ξ, p) yζ(el, ξ, p)
                                    zξ(el, ξ, p) zη(el, ξ, p) zζ(el, ξ, p)])
j(el::AbstractElement, ξ, p) = det([xξ(el, ξ, p) xη(el, ξ, p) xζ(el, ξ, p)
                                    yξ(el, ξ, p) yη(el, ξ, p) yζ(el, ξ, p)
                                    zξ(el, ξ, p) zη(el, ξ, p) zζ(el, ξ, p)])

#### Transformations #### 

x(el::Union{Triangle, Wedge}, ξ, p) = sum(φ(el, ξ, i)*p[i, :] for i ∈ axes(p, 1))
x(el::Line, ξ, p) = sum(φ(el, ξ, i)*p[i] for i ∈ eachindex(p))

function ξ(el::Line, x, p)
    a = (p[2] - p[1])/2
    b = (p[1] + p[2])/2
    return (x - b)/a
end
function ξ(el::Triangle, x, p)
    A = [p[j+1, i] - p[1, i] for i=1:2, j=1:2]
    b = p[1, :]
    return A\(x .- b)
end
function ξ(el::Wedge, x, p)
    ζ = (ξ(Line(order=1), x[3], [p[1, 3], p[4, 3]]) + 1)/2
    ξη = ξ(Triangle(order=1), x[1:2], p[1:3, 1:2])
    return [ξη[1], ξη[2], ζ]
end