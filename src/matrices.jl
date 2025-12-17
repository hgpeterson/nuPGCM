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
    X = fe_data.spaces.X
    Y = fe_data.spaces.Y
    dΩ = fe_data.mesh.dΩ
    α²ε² = params.α^2*params.ε^2
    f = params.f

    # bilinear form
    a((u, p), (v, q)) = bilinear_form((u, p), (v, q), α²ε², f, ν, dΩ; friction_only, frictionless_only)
    a_mini((u, ub, p), (v, vb, q)) = a((u+ub, p), (v+vb, q))

    # assemble 
    @time "build inversion system" A = assemble_matrix(a_mini, X, Y)

    return A
end
function bilinear_form((u, p), (v, q), α²ε², f, ν, dΩ; friction_only, frictionless_only)
    σ = Gridap.symmetric_gradient
    return ∫( 2*α²ε²*ν*σ(u)⊙σ(v) - (∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
#     # for general ν, need full stress tensor
#     if friction_only
#         return ∫( α²ε²*(ν*(∇(ux)⋅∇(vx) + ∂x(ux)*∂x(vx) +                 ∂x(uy)*∂y(vx) +                 ∂x(uz)*∂z(vx))) +
#                   α²ε²*(ν*(              ∂y(ux)*∂x(vy) +   ∇(uy)⋅∇(vy) + ∂y(uy)*∂y(vy) +                 ∂y(uz)*∂z(vy))) +
#                   α²ε²*(ν*(              ∂z(ux)*∂x(vz) +                 ∂z(uy)*∂y(vz) +   ∇(uz)⋅∇(vz) + ∂z(uz)*∂z(vz))) )dΩ
#     elseif frictionless_only
#         return ∫( -(f*uy*vx) + ∂x(p)*vx +
#                     f*ux*vy  + ∂y(p)*vy +
#                                ∂z(p)*vz +
#                     ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
#     else
#         return ∫( α²ε²*(ν*(∇(ux)⋅∇(vx) + ∂x(ux)*∂x(vx) +                 ∂x(uy)*∂y(vx) +                 ∂x(uz)*∂z(vx))) - f*uy*vx + ∂x(p)*vx +
#                   α²ε²*(ν*(              ∂y(ux)*∂x(vy) +   ∇(uy)⋅∇(vy) + ∂y(uy)*∂y(vy) +                 ∂y(uz)*∂z(vy))) + f*ux*vy + ∂y(p)*vy +
#                   α²ε²*(ν*(              ∂z(ux)*∂x(vz) +                 ∂z(uy)*∂y(vz) +   ∇(uz)⋅∇(vz) + ∂z(uz)*∂z(vz))) +           ∂z(p)*vz +
#                   ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
#     end
end
# function bilinear_form((ux, uy, uz, p), (vx, vy, vz, q), α²ε², f, ν::Real, dΩ; friction_only, frictionless_only)
#     # since ν is constant, we can just use the Laplacian here
#     if friction_only
#         return ∫( α²ε²*(ν*∇(ux)⋅∇(vx)) +
#                   α²ε²*(ν*∇(uy)⋅∇(vy)) +
#                   α²ε²*(ν*∇(uz)⋅∇(vz)) )dΩ
#     elseif frictionless_only
#         return ∫( -(f*uy*vx) + ∂x(p)*vx +
#                     f*ux*vy  + ∂y(p)*vy +
#                                ∂z(p)*vz +
#                     ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
#     else
#         return ∫( α²ε²*(ν*∇(ux)⋅∇(vx)) - f*uy*vx + ∂x(p)*vx +
#                   α²ε²*(ν*∇(uy)⋅∇(vy)) + f*ux*vy + ∂y(p)*vy +
#                   α²ε²*(ν*∇(uz)⋅∇(vz)) +           ∂z(p)*vz +
#                   ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
#     end
# end

"""
    B = build_B_inversion(fe_data::FEData, params::Parameters)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(fe_data::FEData, params::Parameters)
    # unpack
    V  = fe_data.spaces.Y[1]
    VB = fe_data.spaces.Y[2]
    B_trial = fe_data.spaces.B
    dΩ = fe_data.mesh.dΩ
    α = params.α

    # bilinear form
    a(b, v) = ∫( 1/α*(b*(z⃗⋅v)) )dΩ

    # assemble
    B0 = assemble_matrix(a, B_trial, V) 
    BB = assemble_matrix(a, B_trial, VB) 
    B = [B0; BB]

    # convert to N × nb matrix
    nu, nub, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + nub + np
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
    V  = fe_data.spaces.Y[1]
    VB = fe_data.spaces.Y[2]
    b_diri = fe_data.spaces.b_diri
    dΓ = fe_data.mesh.dΓ
    dΩ = fe_data.mesh.dΩ
    α = params.α
    τˣ = forcings.τˣ
    τʸ = forcings.τʸ

    # allocate vector of length N
    nu, nub, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + nub + np
    b = zeros(N)

    # linear form
    l(v) = ∫( α*(τˣ*(x⃗⋅v) + τʸ*(y⃗⋅v)) )dΓ + # b.c. is α²ε²ν∂z(u) = ατ
           ∫( 1/α*(b_diri*(z⃗⋅v)) )dΩ        # correction due to Dirichlet boundary condition

    # assemble
    b[1:nu] .= assemble_vector(l, V)
    b[nu+1:nu+nub] .= assemble_vector(l, VB)

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
function build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings)
    @time "build advection system" A_adv = build_advection_matrix(fe_data)
    @time "build hdiffusion system" A_hdiff, B_hdiff, b_hdiff = build_hdiffusion_system(fe_data, params, forcings, forcings.κₕ)
    @time "build vdiffusion system" A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(fe_data, params, forcings, forcings.κᵥ)
    return A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff
end

"""
    A = build_advection_matrix(fe_data::FEData)

Assemble the LHS matrix `A` for the advection part of the evolution problem.

It turns out that the advection matrix is just the mass matrix.
"""
function build_advection_matrix(fe_data::FEData)
    B = fe_data.spaces.B
    D = fe_data.spaces.D
    dΩ = fe_data.mesh.dΩ

    a(b, d) = ∫( b*d )dΩ
    A = assemble_matrix(a, B, D)
    return A
end

"""
    A, B, b = build_hdiffusion_system(fe_data::FEData, params::Parameters, κₕ)

Assemble the matrices for the horizontal diffusion part of the evolution problem.

See also [`build_diffusion_system`](@ref).
"""
function build_hdiffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κₕ)
    return build_diffusion_system(fe_data, params, forcings::Forcings, κₕ, :horizontal)
end

"""
    A, B, b = build_vdiffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κᵥ)

Assemble the matrices for the vertical diffusion part of the evolution problem.

See also [`build_diffusion_system`](@ref).
"""
function build_vdiffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κᵥ)
    return build_diffusion_system(fe_data, params, forcings::Forcings, κᵥ, :vertical)
end

"""
    A, B, b = build_diffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κ, direction::Symbol)

Assemble the matrices for the diffusion part of the evolution problem.

We use the Crank-Nicolson scheme, i.e., 
```math
A b^{n+1} = B b^n + b
```
where ``A = M + θ K`` and ``B = M - θ K`` with ``θ = Δt/4 α² ε² / μϱ`` and `M` and `K` being the mass 
and stiffness matrices, respectively.

`direction` must be either `:horizontal` or `:vertical`.
"""
function build_diffusion_system(fe_data::FEData, params::Parameters, forcings::Forcings, κ, direction::Symbol)
    if direction != :horizontal && direction != :vertical
        throw(ArgumentError("direction must be :horizontal or :vertical"))
    end

    B = fe_data.spaces.B
    D = fe_data.spaces.D
    dΩ = fe_data.mesh.dΩ
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    b_diri = fe_data.spaces.b_diri

    # coefficient for diffusion step (Δt/2 for Crank-Nicolson and Δt/2 for Strang splitting makes Δt/4)
    θ = Δt/4 * α^2 * ε^2 / μϱ

    function a_lhs(b, d)
        if direction == :horizontal
            return ∫( b*d + θ*(κ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d))) )dΩ
        else
            return ∫( b*d + θ*(κ*∂z(b)*∂z(d)) )dΩ
        end
    end

    function a_rhs(b, d)
        if direction == :horizontal
            return ∫( b*d - θ*(κ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d))) )dΩ
        else
            return ∫( b*d - θ*(κ*∂z(b)*∂z(d)) )dΩ
        end
    end

    A = assemble_matrix(a_lhs, B, D)
    B = assemble_matrix(a_rhs, B, D)
    b   = assemble_vector(d -> a_rhs(b_diri, d), D)
    b .-= assemble_vector(d -> a_lhs(b_diri, d), D)

    if direction == :vertical
        # vector for nonzero N² (no Δt/2 for Crank-Nicolson here since it's fully on the RHS)
        N² = params.N²
        l(d) = ∫( -2*θ*N²*(κ*∂z(d)) )dΩ
        b .+= assemble_vector(l, D)

        # see multiple-dispatched functions below
        add_surface_flux!(b, forcings.b_surface_bc, params, κ, fe_data.mesh.dΓ, D)
    end

    return A, B, b
end

function add_surface_flux!(b, bc::SurfaceFluxBC, params::Parameters, κ, dΓ, D)
    N² = params.N²
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    # b.c. is α²ε²/μϱ κ ∂z(b) = α*F (Δt/2 because of Strang split)
    l(d) = ∫( Δt/2 * (α*(bc.flux*d) - α^2*ε^2/μϱ*N²*(κ*d)) )dΓ  
    b .+= assemble_vector(l, D)
    return b
end

function add_surface_flux!(b, bc::SurfaceDirichletBC, args...)
    # `bc` is not a flux condition, continue
    return b
end