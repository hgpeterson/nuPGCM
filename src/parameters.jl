struct Parameters{T<:Real}
    ε::T
    α::T
    μϱ::T
    N²::T
    Δt::T

    # inner constructor to ensure all parameters are of the same type
    function Parameters(args...)
        args = promote(args...)
        T = typeof(args[1])
        return new{T}(args...)
    end
end

function Base.show(io::IO, params::Parameters)
    println(summary(params))
    println(io, @sprintf("├── ε  = %1.1e", params.ε))
    println(io, @sprintf("├── α  = %1.1e", params.α))
    println(io, @sprintf("├── μϱ = %1.1e", params.μϱ))
    println(io, @sprintf("├── N² = %1.1e", params.N²))
    println(io, @sprintf("└── Δt = %1.1e", params.Δt))
end


