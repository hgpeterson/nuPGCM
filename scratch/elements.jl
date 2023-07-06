abstract type AbstractElement end

struct Wedge{IN<:Integer, FM<:AbstractMatrix, IM<:AbstractMatrix} <: AbstractElement
    order::IN
    p_ref::FM
    C::IM
end

function Wedge(; order)
    if order == 1
        p_ref = [ 0  0  0
                  1  0  0
                  0  1  0
                  0  0  1
                  1  0  1
                  0  1  1]
        C = [1	0	0	0	0	0
            -1	1	0	0	0	0
            -1	0	1	0	0	0
            -1	0	0	1	0	0
            1	-1	0	-1	1	0
            1	0	-1	-1	0	1]
    elseif order == 2
        p_ref = [0   0   0
                1   0   0
                0   1   0
                0   0   1
                1   0   1
                0   1   1
                0.5 0   0
                0.5 0.5 0
                0   0.5 0
                0.5 0   1
                0.5 0.5 1
                0   0.5 1]
        C = [1	0	0	0	0	0	0	0	0	0	0	0
            -3	-1	0	0	0	0	4	0	0	0	0	0
            -3	0	-1	0	0	0	0	0	4	0	0	0
            -1	0	0	1	0	0	0	0	0	0	0	0
            3	1	0	-3	-1	0	-4	0	0	4	0	0
            4	0	0	0	0	0	-4	4	-4	0	0	0
            3	0	1	-3	0	-1	0	0	-4	0	0	4
            -4	0	0	4	0	0	4	-4	4	-4	4	-4
            2	2	0	0	0	0	-4	0	0	0	0	0
            2	0	2	0	0	0	0	0	-4	0	0	0
            -2	-2	0	2	2	0	4	0	0	-4	0	0
            -2	0	-2	2	0	2	0	0	4	0	0	-4]
        # p_ref = [0   0   0
        #         1   0   0
        #         0   1   0
        #         0   0   1
        #         1   0   1
        #         0   1   1
        #         0.5 0   0
        #         0.5 0.5 0
        #         0   0.5 0
        #         0.5 0   1
        #         0.5 0.5 1
        #         0   0.5 1
        #         0   0   0.5
        #         1   0   0.5
        #         0   1   0.5]
        # C = [1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
        #     -3	-1	0	0	0	0	4	0	0	0	0	0	0	0	0
        #     -3	0	-1	0	0	0	0	0	4	0	0	0	0	0	0
        #     -3	0	0	-1	0	0	0	0	0	0	0	0	4	0	0
        #     5	-1	0	-1	-3	0	-4	0	0	4	0	0	-4	4	0
        #     4	0	0	0	0	0	-4	4	-4	0	0	0	0	0	0
        #     5	0	-1	-1	0	-3	0	0	-4	0	0	4	-4	0	4
        #     -4	0	0	4	0	0	4	-4	4	-4	4	-4	0	0	0
        #     2	2	0	0	0	0	-4	0	0	0	0	0	0	0	0
        #     2	0	2	0	0	0	0	0	-4	0	0	0	0	0	0
        #     2	0	0	2	0	0	0	0	0	0	0	0	-4	0	0
        #     -2	-2	0	2	2	0	4	0	0	-4	0	0	0	0	0
        #     -2	0	-2	2	0	2	0	0	4	0	0	-4	0	0	0
        #     -2	2	0	-2	2	0	0	0	0	0	0	0	4	-4	0
        #     -2	0	2	-2	0	2	0	0	0	0	0	0	4	0	-4]
    else
        error("Unsupported order")
    end

    return Wedge(order, p_ref, C)
end

function φ(w::Wedge, ξ, i)
    if w.order == 1
        return ([1 ξ[1] ξ[2] ξ[3] ξ[1]*ξ[3] ξ[2]*ξ[3]]*w.C[:, i])[1]
    else w.order == 2
        return ([1 ξ[1] ξ[2] ξ[3] ξ[1]*ξ[3] ξ[1]*ξ[2] ξ[2]*ξ[3] ξ[1]*ξ[2]*ξ[3] ξ[1]^2 ξ[2]^2 ξ[1]^2*ξ[3] ξ[2]^2*ξ[3]]*w.C[:, i])[1]
        # return ([1 ξ[1] ξ[2] ξ[3] ξ[1]*ξ[3] ξ[1]*ξ[2] ξ[2]*ξ[3] ξ[1]*ξ[2]*ξ[3] ξ[1]^2 ξ[2]^2 ξ[3]^2 ξ[1]^2*ξ[3] ξ[2]^2*ξ[3] ξ[1]*ξ[3]^2 ξ[2]*ξ[3]^2]*w.C[:, i])[1]
    end
end
function φξ(w::Wedge, ξ, i)
    if w.order == 1
        return ([0 1 0 0 ξ[3] 0]*w.C[:, i])[1]
    else w.order == 2
        return ([0 1 0 0 ξ[3] ξ[2] 0 ξ[2]*ξ[3] 2*ξ[1] 0 2*ξ[1]*ξ[3] 0]*w.C[:, i])[1]
        # return ([0 1    0    0    ξ[3]      ξ[2]      0         ξ[2]*ξ[3]      2*ξ[1] 0      0      2*ξ[1]*ξ[3] 0           ξ[3]^2      0]*w.C[:, i])[1]
    end
end
function φη(w::Wedge, ξ, i)
    if w.order == 1
        return ([0 0 1 0 0 ξ[3]]*w.C[:, i])[1]
    else w.order == 2
        return ([0 0 1 0 0 ξ[1] ξ[3] ξ[1]*ξ[3] 0 2*ξ[2] 0 2*ξ[2]*ξ[3]]*w.C[:, i])[1]
        # return ([0 0    1    0    0         ξ[1]      ξ[3]      ξ[1]*ξ[3]      0      2*ξ[2] 0      0           2*ξ[2]*ξ[3] 0           ξ[3]^2]*w.C[:, i])[1]
    end
end
function φζ(w::Wedge, ξ, i)
    if w.order == 1
        return ([0 0 0 1 ξ[1] ξ[2]]*w.C[:, i])[1]
    else w.order == 2
        return ([0 0 0 1 ξ[1] 0 ξ[2] ξ[1]*ξ[2] 0 0 ξ[1]^2 ξ[2]^2]*w.C[:, i])[1]
        # return ([0 0    0    1    ξ[1]      0         ξ[2]      ξ[1]*ξ[2]      0      0      2*ξ[3] ξ[1]^2      ξ[2]^2      2*ξ[1]*ξ[3] 2*ξ[2]*ξ[3]]*w.C[:, i])[1]
    end
end

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
