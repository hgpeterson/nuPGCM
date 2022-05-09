# function evaluate(m::ModelSetup3DPG, u)
#     return evaluate(u, m.p₀, m.p, m.t, m.C₀)
# end

function ∂ξ(m, u, p₀)
    # find triangle p₀ is in
    k = get_tri(p₀, m.p, m.t)

    # evaluate there
    return ∂ξ(m, u, p₀, k)
end
function ∂ξ(m, u, p₀, k)
    # sum weighted combinations of c₂
    return dot(u[m.t[k, :]], m.C₀[k, 2, :])
end

function ∂η(m, u, p₀)
    # find triangle p₀ is in
    k = get_tri(p₀, m.p, m.t)

    # evaluate there
    return ∂η(m, u, p₀, k)
end
function ∂η(m, u, p₀, k)
    # sum weighted combinations of c₃
    return dot(u[m.t[k, :]], m.C₀[k, 3, :])
end

function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, p₀)
    # find triangle p₀ is in
    k = get_tri(p₀, m.p, m.t)

    # evaluate there
    return curl(m, u, p₀, k)
end
function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, p₀, k)
    return ∂ξ(m, u[2, :], p₀, k) - ∂η(m, u[1, :], p₀, k)
end