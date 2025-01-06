"""
    x = chebyshev_nodes(n)

Return `n` Chebyshev nodes in the interval `[-1, 0]`.
"""
function chebyshev_nodes(n)
    return ([-cos((i - 1)*π/(n - 1)) for i ∈ 1:n] .- 1)/2
end

"""
    F = trapz(f, z)

Integrate the function `f` over the grid `z` using the trapezoidal rule.
""" 
function trapz(f, z)
    F = 0.0
    i = 1
    while i < length(z)
        # skip NaNs
        if isnan(f[i])
            i += 1
            continue
        end
        j = i + 1
        while isnan(f[j]) && j < length(z)
            j += 1
        end
        if isnan(f[j])
            break
        end

        # area of trapezoid
        F += 0.5*(z[j] - z[i])*(f[j] + f[i])

        # next
        i = j
    end
    return F
end

"""
    F = cumtrapz(f, z)

Cumulatively integrate the function `f` over the grid `z` using the trapezoidal rule.
"""
function cumtrapz(f, z)
    F = zeros(length(z))
    for i ∈ 2:length(z)
        F[i] = F[i-1] + 0.5*(z[i] - z[i-1])*(f[i] + f[i-1])
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