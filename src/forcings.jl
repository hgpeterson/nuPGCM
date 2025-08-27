struct Forcings{KH, KV, KV0, TX, TY, B0, M}
    κₕ::KH        # horizontal diffusivity
    κᵥ::KV        # actual vertical diffusivity (enhanced in unstable regions)
    κᵥ₀::KV0      # goal vertical diffusivity (if no unstable regions)
    κ_conv::Real  # vertical diffusivity in convective regions
    τˣ::TX        # surface zonal wind stress
    τʸ::TY        # surface meridional wind stress
    b₀::B0        # surface buoyancy
    Aκ::M         # matrix for updating κᵥ
end

function Base.show(io::IO, f::Forcings)
    println("Forcings:")
    println(io, "├── κₕ ", f.κₕ)
    println(io, "├── κᵥ ", summary(f.κᵥ))
    println(io, "├── κᵥ₀ ", summary(f.κᵥ₀))
    println(io, "├── κ_conv = ", f.κ_conv)
    println(io, "├── τˣ ", f.τˣ)
    println(io, "├── τʸ ", f.τʸ)
      print(io, "└── b₀ ", f.b₀)
end

function Forcings(fe_data::FEData, κₕ, κᵥ, κ_conv, τˣ, τʸ, b₀)
    spaces = fe_data.spaces

    # need to make κᵥ and κᵥ₀ FEFunctions so that we can update
    κᵥ  = interpolate_everywhere(κᵥ, spaces.κ_trial)
    κᵥ₀ = interpolate_everywhere(κᵥ, spaces.κ_trial)
    T = eltype(κᵥ.free_values)

    # LHS matrix for updating κᵥ (just a mass matrix)
    dΩ = fe_data.mesh.dΩ
    a(u, v) = ∫( u*v )dΩ
    Aκ = assemble_matrix(a, spaces.κ_trial, spaces.κ_test)
    Aκ = lu(Aκ)

    return Forcings(κₕ, κᵥ, κᵥ₀, T(κ_conv), τˣ, τʸ, b₀, Aκ)
end

function update_κᵥ!(f::Forcings, fe_data::FEData, b)
    spaces = fe_data.spaces

    # rhs nonzero where ∂z(b) < 0 (i.e. unstable stratification)
    stability(x) = x < 0 ? 1.0 : 0.0
    dΩ = fe_data.mesh.dΩ
    l(v) = ∫( (stability∘∂z(b))*v )dΩ
    y = assemble_vector(l, spaces.κ_test)

    # κᵥ = κ_conv where unstable
    sol = clamp.(f.Aκ\y, 0.0, 1.0)  # have to clamp between 0 and 1 to avoid weird negative values
    where_unstable = sol .== 1.0
    @debug "Updating κᵥ: $(sum(where_unstable))/$(length(sol)) unstable nodes"
    f.κᵥ.free_values .= f.κᵥ₀.free_values  # reset
    f.κᵥ.free_values[where_unstable] .= f.κ_conv
    return f
end