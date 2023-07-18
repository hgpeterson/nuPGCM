"""
    integral = trapz(f, x)

Integrate array `f` over domain `x` using trapezoidal rule.
"""
function trapz(f, x)
    return 0.5*dot((f[1:end-1] .+ f[2:end]), (x[2:end] .- x[1:end-1]))
end

"""
    integral = cumtrapz(f, x)

Cumulatively integrate array `f` over domain `x` using trapezoidal rule.
"""
function cumtrapz(f, x)
    n = size(f, 1)
    y = zeros(n)
    for i=2:n
        y[i] = y[i-1] + 0.5*(f[i] + f[i-1])*(x[i] - x[i-1])
    end
    return y
end

"""
    integral = ref_el_quad(f, el)

Compute integral of `f` over the reference element `el`. 
"""
function ref_el_quad(f::Function, el::AbstractElement)
    return sum(el.quad_wts[i]*f(el.quad_pts[i, :]) for i ∈ eachindex(el.quad_wts))
end