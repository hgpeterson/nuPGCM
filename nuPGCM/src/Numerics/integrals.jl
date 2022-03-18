"""
    integral = trapz(f, x)

Integrate array `f` over domain `x` using trapezoidal rule.
"""
function trapz(f::Array{Float64,1}, x::Array{Float64,1})
    return 0.5*sum((f[1:end-1] .+ f[2:end]).*(x[2:end] .- x[1:end-1]))
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