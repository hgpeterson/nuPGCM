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
    @time "build advection system" A_adv = build_advection_matrix(fe_data)
    @time "build hdiffusion system" A_hdiff, B_hdiff, b_hdiff = build_hdiffusion_system(fe_data, params, forcings, forcings.κₕ; order)
    @time "build vdiffusion system" A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(fe_data, params, forcings, forcings.κᵥ; order)
    return A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff
end

"""
    A = build_advection_matrix(fe_data::FEData)

Assemble the LHS matrix `A` for the advection part of the evolution problem.

It turns out that the advection matrix is just the mass matrix.
"""
function build_advection_matrix(fe_data::FEData)
    B_trial = fe_data.spaces.B_trial
    B_test = fe_data.spaces.B_test
    dΩ = fe_data.mesh.dΩ

    a(b, d) = ∫( b*d )dΩ
    A = assemble_matrix(a, B_trial, B_test)
    return A
end

"""
    A, B, b = build_hdiffusion_system(fe_data::FEData, params::Parameters, κₕ; order)

Assemble the matrices for the horizontal diffusion part of the evolution problem.

See also [`build_diffusion_system`](@ref).
"""
function build_hdiffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κₕ; order)
    return build_diffusion_system(fe_data, params, forcings::Forcings, κₕ, :horizontal; order)
end

"""
    A, B, b = build_vdiffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κᵥ; order)

Assemble the matrices for the vertical diffusion part of the evolution problem.

See also [`build_diffusion_system`](@ref).
"""
function build_vdiffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κᵥ; order)
    return build_diffusion_system(fe_data, params, forcings::Forcings, κᵥ, :vertical; order)
end

"""
    A, B, b = build_diffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κ, direction::Symbol; order)

Assemble the matrices for the diffusion part of the evolution problem.

The linear system is written as
```math
A b^{n+1} = B b^n + b.
```
If `order == 1`, we use backward Euler, so ``A = M + θ K`` and ``B = M`` with ``θ = Δt/2 α²ε²/μϱ`` and `M` and `K` 
being the mass and stiffness matrices, respectively. For `order == 2`, we use Crank-Nicolson, so ``A = M + θ/2 K`` 
and ``B = M - θ/2 K``.

`direction` must be either `:horizontal` or `:vertical`.
"""
function build_diffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κ, direction::Symbol; order)
    if direction != :horizontal && direction != :vertical
        throw(ArgumentError("direction must be :horizontal or :vertical"))
    end
    if order != 1 && order != 2
        throw(ArgumentError("order must be 1 or 2"))
    end

    B_trial = fe_data.spaces.B_trial
    B_test = fe_data.spaces.B_test
    dΩ = fe_data.mesh.dΩ
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    b_diri = fe_data.spaces.b_diri

    # coefficient for diffusion step (Δt/2 for Strang splitting)
    θ = Δt/2 * α^2 * ε^2 / μϱ

    function a_lhs(b, d)
        if order == 1
            # backward Euler
            if direction == :horizontal
                return ∫( b*d + θ*(κ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d))) )dΩ
            else
                return ∫( b*d + θ*(κ*∂z(b)*∂z(d)) )dΩ
            end
        else
            # Crank-Nicolson
            if direction == :horizontal
                return ∫( b*d + θ/2*(κ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d))) )dΩ
            else
                return ∫( b*d + θ/2*(κ*∂z(b)*∂z(d)) )dΩ
            end
        end
    end
    function a_rhs(b, d)
        if order == 1
            # backward Euler
            return ∫( b*d )dΩ
        else
            # Crank-Nicolson
            if direction == :horizontal
                return ∫( b*d - θ/2*(κ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d))) )dΩ
            else
                return ∫( b*d - θ/2*(κ*∂z(b)*∂z(d)) )dΩ
            end
        end
    end

    A = assemble_matrix(a_lhs, B_trial, B_test)
    B = assemble_matrix(a_rhs, B_trial, B_test)
    b   = assemble_vector(d -> a_rhs(b_diri, d), B_test)
    b .-= assemble_vector(d -> a_lhs(b_diri, d), B_test)

    if direction == :vertical
        # vector for nonzero N²
        N² = params.N²
        l(d) = ∫( -θ*N²*(κ*∂z(d)) )dΩ
        b .+= assemble_vector(l, B_test)

        # see multiple-dispatched functions below
        add_surface_flux!(b, forcings.b_surface_bc, params, κ, fe_data.mesh.dΓ, B_test)
    end

    return A, B, b
end

function add_surface_flux!(b, bc::SurfaceFluxBC, params::Parameters, κ, dΓ, B_test)
    N² = params.N²
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    # b.c. is α²ε²/μϱ κ ∂z(b) = α*F (Δt/2 because of Strang split)
    l(d) = ∫( Δt/2 * (α*(bc.flux*d) - α^2*ε^2/μϱ*N²*(κ*d)) )dΓ  
    b .+= assemble_vector(l, B_test)
    return b
end

function add_surface_flux!(b, bc::SurfaceDirichletBC, args...)
    # `bc` is not a flux condition, continue
    return b
end