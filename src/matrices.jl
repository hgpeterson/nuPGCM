# unit vectors
x⃗ = VectorValue(1.0, 0.0, 0.0)
y⃗ = VectorValue(0.0, 1.0, 0.0)
z⃗ = VectorValue(0.0, 0.0, 1.0)

# gradients 
∂x(u) = x⃗⋅∇(u)
∂y(u) = y⃗⋅∇(u)
∂z(u) = z⃗⋅∇(u)

"""
    A, B, b = build_inversion_system(fe_data::FEData, params::Parameters, forcings::Forcings) 

Build the matrices and vectors for the inversion problem of the PG equations.
"""
function build_inversion_system(fe_data::FEData, params::Parameters, forcings::Forcings) 
    A_inversion = build_A_inversion(fe_data, params, forcings.ν)
    B_inversion = build_B_inversion(fe_data, params)
    b_inversion = build_b_inversion(fe_data, params, forcings)
    return A_inversion, B_inversion, b_inversion
end

"""
    A = build_A_inversion(fe_data::FEData, params::Parameters, ν)

Assemble the LHS matrix `A` for the inversion problem. 
"""
function build_A_inversion(fe_data::FEData, params::Parameters, ν; friction_only=false, frictionless_only=false) 
    # unpack
    X_trial = fe_data.spaces.X_trial
    X_test = fe_data.spaces.X_test
    dΩ = fe_data.mesh.dΩ
    α²ε² = params.α^2*params.ε^2
    f = params.f

    # bilinear form
    a((u, p), (v, q)) = bilinear_form((u, p), (v, q), α²ε², f, ν, dΩ; friction_only, frictionless_only)

    # assemble 
    @time "build inversion system" A = assemble_matrix(a, X_trial, X_test)

    return A
end
function bilinear_form((u, p), (v, q), α²ε², f, ν, dΩ; friction_only, frictionless_only)
    σ = Gridap.symmetric_gradient
    # for general ν, need full stress tensor
    if friction_only
        return ∫( 2*α²ε²*(ν*σ(u)⊙σ(v)) )*dΩ
    elseif frictionless_only
        return ∫( -(∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    else
        return ∫( 2*α²ε²*(ν*σ(u)⊙σ(v)) - (∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    end
end
function bilinear_form((u, p), (v, q), α²ε², f, ν::Real, dΩ; friction_only, frictionless_only)
    # since ν is constant, we can just use the Laplacian here
    if friction_only
        return ∫( α²ε²*(ν*∇(u)⊙∇(v)) )*dΩ
    elseif frictionless_only
        return ∫( -(∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    else
        return ∫( α²ε²*(ν*∇(u)⊙∇(v)) - (∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    end
end

"""
    B = build_B_inversion(fe_data::FEData, params::Parameters)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(fe_data::FEData, params::Parameters)
    # unpack
    U_test = fe_data.spaces.X_test[1]
    B_trial = fe_data.spaces.B_trial
    dΩ = fe_data.mesh.dΩ
    α = params.α

    # bilinear form
    a(b, v) = ∫( 1/α*(b*(z⃗⋅v)) )dΩ

    # assemble
    B = assemble_matrix(a, B_trial, U_test) 

    # convert to N × nb matrix
    nu, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + np
    I, J, V = findnz(B)
    B = sparse(I, J, V, N, nb)

    return B
end

"""
    b = build_b_inversion(mesh::FEData, params::Parameters, forcings::Forcings)

Assemble the RHS vector for the inversion problem.
"""
function build_b_inversion(fe_data::FEData, params::Parameters, forcings::Forcings)
    # unpack
    U_test  = fe_data.spaces.X_test[1]
    b_diri = fe_data.spaces.b_diri
    dΓ = fe_data.mesh.dΓ
    dΩ = fe_data.mesh.dΩ
    α = params.α
    τˣ = forcings.τˣ
    τʸ = forcings.τʸ

    # allocate vector of length N
    nu, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + np
    b = zeros(N)

    # linear form
    l(v) = ∫( α*(τˣ*(x⃗⋅v) + τʸ*(y⃗⋅v)) )dΓ + # b.c. is α²ε²ν∂z(u) = ατ
           ∫( 1/α*(b_diri*(z⃗⋅v)) )dΩ        # correction due to Dirichlet boundary condition

    # assemble
    b[1:nu] .= assemble_vector(l, U_test)

    return b
end


"""
    A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = 
        build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings)

Build the matrices for the evolution problem of the PG equations.

The evolution equation is written as
```math
μϱ ( ∂ₜb + u·∇b ) = α²ε² [ ∇ₕ·(κₕ∇ₕb) + ∂z(κᵥ∂z b) ]
```
We use Strang splitting to split the evolution into an advection step and two diffusion steps (horizontal and vertical).
The linear systems are written as
```math
A_adv b^{n+1} = b^n + F_adv
A_hdiff b^{n+1} = B_hdiff b^n + b_hdiff
A_vdiff b^{n+1} = B_vdiff b^n + b_vdiff
```

See also [`build_advection_matrix`](@ref), [`build_hdiffusion_system`](@ref), [`build_vdiffusion_system`](@ref).
"""
function build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings; order)
    if order != 1
        throw(ArgumentError("order $order not yet implemented"))
    end

    @time "build evolution system" begin

    # unpack
    B_trial = fe_data.spaces.B_trial
    B_test = fe_data.spaces.B_test
    b_diri = fe_data.spaces.b_diri
    dΩ = fe_data.mesh.dΩ
    κₕ = forcings.κₕ
    κᵥ = forcings.κᵥ

    # bilinear forms
    aₘ(b, d) = ∫( b*d )dΩ
    aₕ(b, d) = ∫( κₕ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d)) )dΩ
    aᵥ(b, d) = ∫( κᵥ*∂z(b)*∂z(d) )dΩ

    M = assemble_matrix(aₘ, B_trial, B_test)
    Kₕ = assemble_matrix(aₕ, B_trial, B_test)
    Kᵥ = assemble_matrix(aᵥ, B_trial, B_test)

    # rhs vector for nonzero N²
    N² = params.N²
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    l(d) = ∫( -Δt * α^2*ε^2/μϱ * N² * (κᵥ*∂z(d)) )dΩ
    rhs = assemble_vector(l, B_test)

    # rhs vector for surface flux (see multiple-dispatched functions below)
    add_surface_flux!(rhs, forcings.b_surface_bc, params, κᵥ, fe_data.mesh.dΓ, B_test)

    # correction due to Dirichlet b.c.
    rhs .+= assemble_vector(d -> aₘ(b_diri, d), B_test)

    end
    return M, Kₕ, Kᵥ, rhs
end

function add_surface_flux!(rhs, bc::SurfaceFluxBC, params::Parameters, κ, dΓ, B_test)
    N² = params.N²
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    # b.c. is α²ε²/μϱ κ ∂z(b) = α*F
    l(d) = ∫( Δt * (α*(bc.flux*d) - α^2*ε^2/μϱ * N² * (κ*d)) )dΓ  
    rhs .+= assemble_vector(l, B_test)
    return rhs
end

function add_surface_flux!(rhs, bc::SurfaceDirichletBC, args...)
    # `bc` is not a flux condition, continue
    return rhs
end