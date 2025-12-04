#### Parameters type ####

struct Parameters{T<:Real, F, H}
    ε::T      # Ekman number √(ν₀ / (f₀H₀²))
    α::T      # aspect ratio (H₀ / L₀)
    μϱ::T     # Prandtl (ν₀ / κ₀) times Burger number (N₀²H₀² / f₀²L₀²)
    N²::T     # background stratification (nondimensional)
    Δt::T     # timestep
    f::F      # Coriolis parameter (function of x)
    H::H      # Depth (function of x)

    # inner constructor to ensure all parameters are of the same type
    function Parameters(ε, α, μϱ, N², Δt, f, H)
        args = promote(ε, α, μϱ, N², Δt)
        T = typeof(args[1])
        f_type = typeof(f)
        H_type = typeof(H)
        return new{T, f_type, H_type}(args..., f, H)
    end
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
    println(io, @sprintf("├── Δt = %1.1e", params.Δt))
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

function κᵥ_convection(conv_param::ConvectionParameterization, κᵥ, αbz)
    κᶜ = conv_param.κᶜ
    N²min = conv_param.N²min
    return κᵥ + κᶜ*(1 + tanh∘(-(αbz)/N²min))/2
end

#### EddyParameterization type ####

struct EddyParameterization{F, T}
    f::F      # Coriolis
    N²min::T  # minimum stratification α*∂z(b) before eddy parameterization starts tapering off
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
          print(io, @sprintf("└── N²min = %1.1e", eddy_param.N²min))
    else
        print(io, " off")
    end
end

function EddyParameterization(; f, N²min)
    return EddyParameterization(f, N²min, true)
end

function ν_eddy(eddy_param::EddyParameterization, αbz)
    f = eddy_param.f
    N²min = eddy_param.N²min
    return f * (f / (sqrt∘(N²min +  αbz * αbz)))
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
        eddy_param = EddyParameterization(0, 0, false)
    end
    return Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc, conv_param, eddy_param)
end

function κᵥ_convection(forcings::Forcings, αbz)
    return κᵥ_convection(forcings.conv_param, forcings.κᵥ, αbz)
end