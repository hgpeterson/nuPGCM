"""
    x, z = transform_to_physical(x′, z′, θ)

Tranform from rotated to physical coordinates.

The transformation reads:
```math
x = x′\\cos(θ) - z′\\sin(θ) \\quad \\text{and} \\quad z = x′\\sin(θ) + z′\\cos(θ)
```
"""
function transform_to_physical(x′, z′, θ)
    return @. x′*cos(θ) - z′*sin(θ), 
           @. x′*sin(θ) + z′*cos(θ)
end

"""
    x′, z′ = transform_to_rotated(x, z, θ)

Tranform from physical to rotated coordinates.

The transformation reads:
```math
x′ = x\\cos(θ) + z\\sin(θ) \\quad \\text{and} \\quad z′ = -x\\sin(θ) + z\\cos(θ)
```
"""
function transform_to_rotated(x, z, θ)
    return @.  x*cos(θ) + z*sin(θ), 
           @. -x*sin(θ) + z*cos(θ)
end

"""
    ν = update_ν!(ν, b, params; threshold=1e2)

Update turbulent viscosity `ν` according to eddy parameterization.

In physical coordinates, the parameterization is
```math
ν = \\frac{f^2}{α ∂_z b}
```
"""
function update_ν!(ν, b, params; threshold=1e2)
    f = params.f
    α = params.α
    N² = params.N²
    θ = params.θ
    z = params.z
    ν .= abs.(f^2 ./ ( α * (N² .+ cos(θ)*differentiate(b, z)) ))
    ν[ν .> threshold] .= threshold
    return ν
end