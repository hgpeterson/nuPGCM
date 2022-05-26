"""
    integral = trapz(f, x)

Integrate array `f` over domain `x` using trapezoidal rule.
"""
function trapz(f::Array{Float64,1}, x::Array{Float64,1})
    return 0.5*dot((f[1:end-1] .+ f[2:end]), (x[2:end] .- x[1:end-1]))
end

"""
    integral = cumtrapz(f, x)

Cumulatively integrate array `f` over domain `x` using trapezoidal rule.
"""
function cumtrapz(f::Array{Float64,1}, x::Array{Float64,1})
    y = zeros(size(f, 1))
    for i=2:size(f, 1)
        y[i] = y[i-1] + 0.5*(f[i] + f[i-1])*(x[i] - x[i-1])
    end
    return y
end

"""
    integral = gaussian_quad2(f, a, b)

Compute ∫ f(x) dx on [a, b] using second order gaussian quadrature.
"""
function gaussian_quad2(f::Function, a::Real, b::Real)
    # integration points 
    x = @. (b - a)/2*[-1/sqrt(3), 1/sqrt(3)] + (a + b)/2

    # weights are both 1
    return (b - a)/2*(f(x[1]) + f(x[2]))
end

"""
    integral = gaussian_quad2(f, p)

Compute ∫ f(x, y) dA over a triangle defined by the points `p` using second 
order gaussian quadrature.
"""
function gaussian_quad2(f::Function, p::Array{<:Real,2})
    # area of triangle
    area = tri_area(p)

    # integration points (rows: point number, columns: x, y)
	x = p/2 .+ 1/6*sum(p, dims=1)

    # weights are all 1/3
    return area/3 * (f(x[1, 1], x[1, 2]) + f(x[2, 1], x[2, 2]) + f(x[3, 1], x[3, 2]))
end
