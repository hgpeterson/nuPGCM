function evaluate(m::ModelSetup3DPG, u)
    return evaluate(u, m.p₀, m.p, m.t, m.C₀)
end

function ∂ξ(u, p₀, p, t, C₀)
    # find triangle p₀ is in
    k₀ = get_tri(p₀, p, t)

    # evaluate there
    return ∂ξ(u, p₀, p, t, C₀, k₀)
end
function ∂ξ(u, p₀, p, t, C₀, k₀)
    # sum weighted combinations of c₂
    return dot(u[t[k₀, :]], C₀[k₀, 2, :])
end
function ∂ξ(m::ModelSetup3DPG, u)
    return ∂ξ(u, m.p₀, m.p, m.t, m.C₀)
end

function ∂η(u, p₀, p, t, C₀)
    # find triangle p₀ is in
    k₀ = get_tri(p₀, p, t)

    # evaluate there
    return ∂η(u, p₀, p, t, C₀, k₀)
end
function ∂η(u, p₀, p, t, C₀, k₀)
    # sum weighted combinations of c₃
    return dot(u[t[k₀, :]], C₀[k₀, 3, :])
end
function ∂η(m::ModelSetup3DPG, u)
    return ∂η(u, m.p₀, m.p, m.t, m.C₀)
end

function ∇×(m::ModelSetup3DPG, u::AbstractArray{<:Real,2})
    return ∂ξ(m, u[2, :]) - ∂η(m, u[1, :])
end