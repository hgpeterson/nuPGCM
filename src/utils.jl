"""
    x = chebyshev_nodes(n)

Return `n` Chebyshev nodes in the interval `[-1, 1]`.
"""
function chebyshev_nodes(n)
    return [-cos((i - 1)*π/(n - 1)) for i ∈ 1:n]
end

"""
    F = trapz(f, z)

Integrate the function `f` over the grid `z` using the trapezoidal rule.
""" 
function trapz(f, z)
    F = 0.0
    for i ∈ 1:length(z) - 1
        F += 0.5*(z[i + 1] - z[i])*(f[i + 1] + f[i])
    end
    return F
end

"""
    hrs, mins, secs = hrs_mins_secs(seconds)

Converts a number of seconds into hours, minutes, and seconds.
"""
function hrs_mins_secs(seconds)
    return seconds ÷ 3600, (seconds % 3600) ÷ 60, seconds % 60
end

"""
    u_max = nan_max(u)

Returns the maximum value of `u`, ignoring `NaN`s.
"""
function nan_max(u)
    return maximum(i -> isnan(u[i]) ? -Inf : u[i], 1:length(u))
end

"""
    u_min = nan_min(u)  

Returns the minimum value of `u`, ignoring `NaN`s.
"""
function nan_min(u)
    return minimum(i -> isnan(u[i]) ? Inf : u[i], 1:length(u))
end

"""
    s = sci_notation(x)

Returns a string representation of `x` in scientific notation.
"""
function sci_notation(x)
    if x == 0
        return "0.0"
    end
    exp = floor(log10(abs(x)))
    mant = x / 10^exp
    return @sprintf("%1.1f \\times 10^{%d}", mant, exp)
end