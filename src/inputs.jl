struct Parameters{T<:Real, F, H}
    ε::T   # Ekman number √(ν₀ / (f₀H₀²))
    α::T   # aspect ratio (H₀ / L₀)
    μϱ::T  # Prandtl (ν₀ / κ₀) times Burger number (N₀²H₀² / f₀²L₀²)
    N²::T  # background stratification (nondimensional)
    Δt::T  # timestep
    κᶜ::T  # vertical diffusivity in convective regions
    f::F   # Coriolis parameter (function of x)
    H::H   # Depth (function of x)

    # inner constructor to ensure all parameters are of the same type
    function Parameters(ε, α, μϱ, N², Δt, κ_conv, f, H)
        args = promote(ε, α, μϱ, N², Δt, κ_conv)
        T = typeof(args[1])
        f_type = typeof(f)
        H_type = typeof(H)
        return new{T, f_type, H_type}(args..., f, H)
    end
end

function Base.show(io::IO, params::Parameters)
    println(summary(params))
    println(io, @sprintf("├── ε  = %1.1e", params.ε))
    println(io, @sprintf("├── α  = %1.1e", params.α))
    println(io, @sprintf("├── μϱ = %1.1e", params.μϱ))
    println(io, @sprintf("├── N² = %1.1e", params.N²))
    println(io, @sprintf("├── Δt = %1.1e", params.Δt))
    println(io, @sprintf("├── κᶜ = %1.1e", params.κᶜ))
    println(io, @sprintf("├── f"))
      print(io, @sprintf("└── H"))
end

struct Forcings{N, KH, KV, TX, TY, B0}
    ν::N              # viscosity
    κₕ::KH            # horizontal diffusivity
    κᵥ::KV            # vertical diffusivity
    τˣ::TX            # surface zonal wind stress
    τʸ::TY            # surface meridional wind stress
    b₀::B0            # surface buoyancy
    convection::Bool  # whether to use convection scheme (default false)
    eddy_param::Bool  # whether to use eddy parameterization (default false)
end

function Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b₀; convection=false, eddy_param=false)
    return Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b₀, convection, eddy_param)
end

function Base.show(io::IO, f::Forcings)
    println("Forcings:")
    println(io, "├── ν")
    println(io, "├── κₕ")
    println(io, "├── κᵥ")
    println(io, "├── τˣ")
    println(io, "├── τʸ")
    println(io, "├── b₀")
    println(io, "├── convection = ", f.convection)
      print(io, "└── eddy_param = ", f.eddy_param)
end