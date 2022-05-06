# function evaluate(m::ModelSetup3DPG, u)
#     return evaluate(u, m.p₀, m.p, m.t, m.C₀)
# end

function ∂ᵢ(u::AbstractArray{<:Real,1}, p₀::AbstractArray{<:Real,1}, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # find triangle p₀ is in
    k = get_tri(p₀, p, t)

    # evaluate there
    return ∂ᵢ(u, p₀, k, p, t, C₀; i)
end
function ∂ᵢ(u::AbstractArray{<:Real,1}, p₀::AbstractArray{<:Real,1}, k::Integer, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # sum weighted combinations of c₂
    return dot(u[t[k, :]], C₀[k, i+1, :])
end
function ∂ᵢ(m::ModelSetup3DPG, u::AbstractArray{<:Real,1}, p₀::AbstractArray{<:Real,1}; i::Integer)
    return ∂ᵢ(u, p₀, m.p, m.t, m.C₀; i)
end
function ∂ᵢ(m::ModelSetup3DPG, u::AbstractArray{<:Real,1}, p₀::AbstractArray{<:Real,1}, k::Integer; i::Integer)
    return ∂ᵢ(u, p₀, k, m.p, m.t, m.C₀; i)
end

function ∂ξ(args...)
    return ∂ᵢ(args...; i=1)
end
function ∂η(args...)
    return ∂ᵢ(args...; i=2)
end

function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, p₀::AbstractArray{<:Real,1})
    # find triangle p₀ is in
    k = get_tri(p₀, m.p, m.t)

    # evaluate there
    return curl(m, u, p₀, k)
end
function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, p₀::AbstractArray{<:Real,1}, k::Integer)
    return ∂ξ(m, u[2, :], p₀, k) - ∂η(m, u[1, :], p₀, k)
end