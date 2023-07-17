struct Grid{V<:AbstractVector}
    x::V
end

struct GridFunc{V<:AbstractVector, G<:Grid}
    vals::V
    g::G
end

function GridFunc(f::Function, g::Grid)
    return GridFunc([f(g.x[i]) for i ∈ eachindex(g.x)], g)
end

function (f::GridFunc)(x)
    return f.vals[argmin(abs.(x .- f.g.x))]
end

function test()
    g = Grid(-1:0.1:1)
    f = GridFunc(x -> x^2, g)
    return f(0)
end