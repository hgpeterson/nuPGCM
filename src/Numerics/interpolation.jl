"""
    y₀ = lerp(x, y, x₀)

Linear interpolation through points (xᵢ, yᵢ) evaluated at x₀.
"""
function lerp(x, y, x₀)
    # deal with edge cases
    if x₀ < x[1] || x₀ > x[end]
        error("Interpolation point x₀ = $x₀ out of bounds for points x = [$(x[1]) ... $(x[end])].")
    elseif x₀ == x[1]
        return y[1]
    elseif x₀ == x[end]
        return y[end]
    end

    # find index x₀ would be in in x
    i = findall(sortperm([x₀; x]) .== 1)[1]

    # x₁ is left of x₀ and x₂ is right
    x₁ = x[i-1]
    x₂ = x[i]
    y₁ = y[i-1]
    y₂ = y[i]

    # linear interpolation
    return y₁*(x₂ - x₀)/(x₂ - x₁) + y₂*(x₁ - x₀)/(x₁ - x₂)
end