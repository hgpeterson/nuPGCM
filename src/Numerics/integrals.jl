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

function gaussian_quad16(f::Function, p::Array{<:Real,2})
    # x = [0.0571041961  0.06546699455602246
    #      0.2768430136  0.05021012321401679
    #      0.5835904324  0.02891208422223085
    #      0.8602401357  0.009703785123906346
    #      0.0571041961  0.3111645522491480    
    #      0.2768430136  0.2386486597440242    
    #      0.5835904324  0.1374191041243166   
    #      0.8602401357  0.04612207989200404
    #      0.0571041961  0.6317312516508520   
    #      0.2768430136  0.4845083266559759    
    #      0.5835904324  0.2789904634756834    
    #      0.8602401357  0.09363778440799593
    #      0.0571041961  0.8774288093439775    
    #      0.2768430136  0.6729468631859832    
    #      0.5835904324  0.3874974833777692    
    #      0.8602401357  0.1300560791760936]  

    # w = [0.04713673637581137
    #      0.07077613579259895
    #      0.04516809856187617
    #      0.01084645180365496
    #      0.08837017702418863
    #      0.1326884322074010    
    #      0.08467944903812383
    #      0.02033451909634504
    #      0.08837017702418863
    #      0.1326884322074010    
    #      0.08467944903812383
    #      0.02033451909634504
    #      0.04713673637581137
    #      0.07077613579259895
    #      0.04516809856187617
    #      0.01084645180365496]

    x = [0.33333333333333333333  0.33333333333333333333]
    w = [1]

    jacobian = (p[2, 1] - p[1, 1])*(p[3, 2] - p[1, 2]) - (p[3, 1] - p[1, 1])*(p[2, 2] - p[1, 2])

    area = tri_area(p)

    integral = 0
    for i=1:size(w, 1)
        x′ = @. p[1, :] + (p[2, :] - p[1, :])*x[i, 1] + (p[3, :] - p[1, :])*x[i, 2]
        integral += w[i]*f(x′[1], x′[2])
    end

    return area*jacobian*integral
end