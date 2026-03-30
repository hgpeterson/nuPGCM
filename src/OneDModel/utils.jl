"""
    x, z = transform_to_physical(x́, ź, θ)

Tranform from rotated to physical coordinates.

The transformation reads:
```math
x = x́\\cos(θ) - ź\\sin(θ) \\quad \\text{and} \\quad z = x́\\sin(θ) + ź\\cos(θ)
```
"""
function transform_to_physical(x́, ź, θ)
    return x́*cos(θ) - ź*sin(θ), 
           x́*sin(θ) + ź*cos(θ)
end

"""
    x́, ź = transform_to_rotated(x, z, θ)

Tranform from physical to rotated coordinates.

The transformation reads:
```math
x́ = x\\cos(θ) + z\\sin(θ) \\quad \\text{and} \\quad ź = -x\\sin(θ) + z\\cos(θ)
```
"""
function transform_to_rotated(x, z, θ)
    return  x*cos(θ) + z*sin(θ), 
           -x*sin(θ) + z*cos(θ)
end

"""
    ν = update_ν!(ν, b, params; N²min=1e-3, νmin=1, smoothing=10)

Update turbulent viscosity `ν` according to eddy parameterization.

In physical coordinates, the parameterization is
```math
ν = \\frac{f^2}{α ∂_z b}
```
"""
function update_ν!(ν, b, params; N²min=√1e-3, νmin=1, smoothing=10)
    f = params.f
    α = params.α
    N² = params.N²
    θ = params.θ
    z = params.z
    # see `ν_eddy()` in `src/inputs.jl`
    αbz = α * (N² .+ cos(θ)*differentiate(b, z))
    @. ν = f^2 / sqrt(N²min^2 +  αbz^2)  # eddy value
    @. ν = (log(exp(smoothing*νmin) + exp(smoothing*ν)) / smoothing)  # LogSumExp
    return ν
end