"""
    x = chebyshev_nodes(n)

Return `n` Chebyshev nodes in the interval `[-1, 1]`.
"""
function chebyshev_nodes(n)
    return [-cos((i - 1)*π/(n - 1)) for i ∈ 1:n]
end

"""
    hrs, mins, secs = hrs_mins_secs(seconds)

Converts a number of seconds into hours, minutes, and seconds.
"""
function hrs_mins_secs(seconds)
    return seconds ÷ 3600, (seconds % 3600) ÷ 60, seconds % 60
end