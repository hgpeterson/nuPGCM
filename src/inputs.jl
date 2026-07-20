#### Parameters type ####

struct Parameters{T<:Real, F, H}
    ε::T      # Ekman number √(ν₀ / (f₀H₀²))
    α::T      # aspect ratio (H₀ / L₀)
    μϱ::T     # Prandtl (ν₀ / κ₀) times Burger number (N₀²H₀² / f₀²L₀²)
    N²::T     # background stratification (nondimensional)
    f::F      # Coriolis parameter (function of x)
    H::H      # Depth (function of x)
end

function Parameters(; ε, α, μϱ, N², f, H)
    args = promote(ε, α, μϱ, N²)
    return Parameters(args..., f, H)
end

function Base.summary(params::Parameters)
    t = typeof(params)
    return "$(parentmodule(t)).$(nameof(t)){$(t.parameters[1])}"
end
function Base.show(io::IO, params::Parameters)
    println(io, summary(params))
    println(io, @sprintf("├── ε  = %1.1e", params.ε))
    println(io, @sprintf("├── α  = %1.1e", params.α))
    println(io, @sprintf("├── μϱ = %1.1e", params.μϱ))
    println(io, @sprintf("├── N² = %1.1e", params.N²))
    println(io,          "├── f: ", params.f)
      print(io,          "└── H: ", params.H)
end

#### SurfaceBC types ####

abstract type AbstractSurfaceBC end

struct SurfaceDirichletBC{V} <: AbstractSurfaceBC
    value::V
end

function Base.summary(surface_bc::SurfaceDirichletBC)
    t = typeof(surface_bc)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, surface_bc::SurfaceDirichletBC)
    println(io, summary(surface_bc), ":")
      print(io, "└── value: ", surface_bc.value)
end

struct SurfaceFluxBC{F} <: AbstractSurfaceBC
    flux::F
end

function Base.summary(surface_bc::SurfaceFluxBC)
    t = typeof(surface_bc)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, surface_bc::SurfaceFluxBC)
    println(io, summary(surface_bc), ":")
      print(io, "└── flux: ", surface_bc.flux)
end

#### ConvectionParameterization type ####

struct ConvectionParameterization{T}
    κᶜ::T        # vertical diffusivity in convective regions
    N²min::T     # minimum stratification α*∂z(b) before convection starts kicking in
    is_on::Bool
end

function Base.summary(conv_param::ConvectionParameterization)
    t = typeof(conv_param)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, conv_param::ConvectionParameterization)
    print(io, summary(conv_param), ":")
    if conv_param.is_on
        println(io, @sprintf("\n├── κᶜ    = %1.1e", conv_param.κᶜ))
          print(io,   @sprintf("└── N²min = %1.1e", conv_param.N²min))
    else
        print(io, " off")
    end
end

function ConvectionParameterization(; κᶜ, N²min)
    return ConvectionParameterization(κᶜ, N²min, true)
end

"""
    κᶜ = _κ_conv_extra(conv_param, αbz)

Convection augmentation of the vertical diffusivity: κᶜ where the stratification
`αbz` is unstable, tapering to 0 over the scale `N²min`.
"""
function _κ_conv_extra(conv_param::ConvectionParameterization, αbz)
    return conv_param.κᶜ*(1 + tanh(-(αbz)/conv_param.N²min))/2
end

function κᵥ_convection(conv_param::ConvectionParameterization, κᵥ, αbz)
    return κᵥ + _κ_conv_extra(conv_param, αbz)
end

#### EddyParameterization type ####

struct EddyParameterization{F, T}
    f::F      # Coriolis
    N²min::T  # minimum stratification α*∂z(b) before eddy parameterization starts tapering off
    ν_min::T  # minimum eddy viscosity
    smoothing::T  # smoothing parameter for ν_eddy (higher → closer to max(ν, ν_min))
    is_on::Bool
end

function Base.summary(eddy_param::EddyParameterization)
    t = typeof(eddy_param)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, eddy_param::EddyParameterization)
    print(io, summary(eddy_param), ":")
    if eddy_param.is_on
        println(io,        "\n├── f: ", eddy_param.f)
        println(io, @sprintf("├── N²min = %1.1e", eddy_param.N²min))
        println(io, @sprintf("├── ν_min = %1.1e", eddy_param.ν_min))
          print(io, @sprintf("└── smoothing = %1.1e", eddy_param.smoothing))
    else
        print(io, " off")
    end
end

function EddyParameterization(; f, N²min, ν_min=0., smoothing=10.)
    return EddyParameterization(f, N²min, ν_min, smoothing, true)
end

"""
    ν = ν_eddy(eddy_param::EddyParameterization, f, αbz)

Compute ν for eddy parameterization, given the local Coriolis value `f`.

The parameterization reads
```math
ν = f² / (α ∂_z b).
```
We also smoothly limit ν_min ≤ ν ≤ f² / N²min.
"""
function ν_eddy(eddy_param::EddyParameterization, f::Real, αbz)
    (; N²min, ν_min, smoothing) = eddy_param

    # eddy value → f² / (α ∂_z b) for large stratification and → f² / N²min for low stratification
    ν = f * (f / sqrt(N²min^2 + αbz * αbz))

    # LogSumExp converges to max(ν, ν_min) as smoothing → ∞
    # (shift by the max so the exponentials cannot overflow)
    m = max(ν_min, ν)
    return m + log(exp(smoothing*(ν_min - m)) + exp(smoothing*(ν - m))) / smoothing
end

#### Forcings type ####

struct Forcings{N, KH, KV, TX, TY, 
                BC <: AbstractSurfaceBC, 
                CP <: ConvectionParameterization,
                EP <: EddyParameterization}
    ν::N              # viscosity
    κₕ::KH            # horizontal diffusivity
    κᵥ::KV            # vertical diffusivity
    τˣ::TX            # surface zonal wind stress
    τʸ::TY            # surface meridional wind stress
    b_surface_bc::BC  # surface boundary condition for buoyancy
    conv_param::CP    # convection parameterization (default off)
    eddy_param::EP    # eddy parameterization (default off)
end

function Base.summary(forcings::Forcings)
    t = typeof(forcings)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, f::Forcings)
    println(io, summary(f), ":")
    println(io, "├── ν: ", f.ν)
    println(io, "├── κₕ: ", f.κₕ)
    println(io, "├── κᵥ: ", f.κᵥ)
    println(io, "├── τˣ: ", f.τˣ)
    println(io, "├── τʸ: ", f.τʸ)
    println(io, "├── b_surface_bc: ", summary(f.b_surface_bc))
    if f.conv_param.is_on
        println(io, "├── conv_param: ", summary(f.conv_param))
    else
        println(io, "├── conv_param: off")
    end
    if f.eddy_param.is_on
      print(io, "└── eddy_param: ", summary(f.eddy_param))
    else
      print(io, "└── eddy_param: off")
    end
end

function Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param=nothing, eddy_param=nothing)
    if conv_param === nothing
        # by default, no `ConvectionParameterization` (`is_on` == false)
        conv_param = ConvectionParameterization(0, 0, false)
    end
    if eddy_param === nothing
        # by default, no `EddyParameterization` (`is_on` == false)
        eddy_param = EddyParameterization(x -> 0.0, 0.0, 0.0, 10.0, false)
    end
    return Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc, conv_param, eddy_param)
end

function κᵥ_convection(forcings::Forcings, αbz)
    return κᵥ_convection(forcings.conv_param, forcings.κᵥ, αbz)
end