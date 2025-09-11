# gradients 
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

"""
    A, B, b = build_inversion_system(fe_data::FEData, params::Parameters, forcings::Forcings; 
                                     filename="")

Build the matrices and vectors for the inversion problem of the PG equations.
"""
function build_inversion_matrices(fe_data::FEData, params::Parameters, forcings::Forcings; 
                                  filename="")
    A_inversion = build_A_inversion(fe_data, params, forcings; filename)
    B_inversion = build_B_inversion(fe_data, params)
    b_inversion = build_b_inversion(fe_data, params, forcings)
    return A_inversion, B_inversion, b_inversion
end

"""
    A = build_A_inversion(fe_data::FEData, params::Parameters, forcings::Forcings; filename="")

Assemble the LHS matrix `A` for the inversion problem. 
If `filename` is given, the data is saved to a file.
"""
function build_A_inversion(fe_data::FEData, params::Parameters, forcings::Forcings; 
                           filename="")
    # unpack
    X_trial = fe_data.spaces.X_trial
    X_test = fe_data.spaces.X_test
    dΩ = fe_data.mesh.dΩ
    α²ε² = params.α^2*params.ε^2

    # bilinear form
    a((ux, uy, uz, p), (vx, vy, vz, q)) = bilinear_form((ux, uy, uz, p), (vx, vy, vz, q), α²ε², f, ν, dΩ)

    # assemble 
    @time "build A_inversion" A = assemble_matrix(a, X_trial, X_test)

    # save
    if filename != ""
        isfile(filename) && @warn "A_inversion file already exists and will be overwritten." filename
        jldsave(filename; A_inversion=A, params=params, forcings=forcings)
        @info @sprintf("A_inversion saved to '%s' (%.3f GB)", filename, filesize(filename)/1e9)
    end

    return A
end
function bilinear_form((ux, uy, uz, p), (vx, vy, vz, q), α²ε², f, ν, dΩ)
    # for general ν, need full stress tensor
    return ∫( α²ε²*(ν*(∇(ux)⋅∇(vx) + ∂x(ux)*∂x(vx) +                 ∂x(uy)*∂y(vx) +                 ∂x(uz)*∂z(vx))) - f*uy*vx + ∂x(p)*vx +
              α²ε²*(ν*(              ∂y(ux)*∂x(vy) +   ∇(uy)⋅∇(vy) + ∂y(uy)*∂y(vy) +                 ∂y(uz)*∂z(vy))) + f*ux*vy + ∂y(p)*vy +
              α²ε²*(ν*(              ∂z(ux)*∂x(vz) +                 ∂z(uy)*∂y(vz) +   ∇(uz)⋅∇(vz) + ∂z(uz)*∂z(vz))) +           ∂z(p)*vz +
              ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
end
function bilinear_form((ux, uy, uz, p), (vx, vy, vz, q), α²ε², f, ν::Real, dΩ)
    # since ν is constant, we can just use the Laplacian here
    return ∫( α²ε²*(ν*∇(ux)⋅∇(vx)) - f*uy*vx + ∂x(p)*vx +
              α²ε²*(ν*∇(uy)⋅∇(vy)) + f*ux*vy + ∂y(p)*vy +
              α²ε²*(ν*∇(uz)⋅∇(vz)) +           ∂z(p)*vz +
              ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
end

"""
    B = build_B_inversion(fe_data::FEData, params::Parameters)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(fe_data::FEData, params::Parameters)
    # unpack
    W_test = fe_data.spaces.X_test[3]
    B_trial = fe_data.spaces.B_trial
    dΩ = fe_data.mesh.dΩ
    α = params.α

    # bilinear form
    a(b, vz) = ∫( 1/α*(b*vz) )dΩ

    # assemble
    B = assemble_matrix(a, B_trial, W_test)

    # convert to N × nb matrix
    nu, nv, nw, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + nv + nw + np
    I, J, V = findnz(B)
    I .+= nu + nv
    B = sparse(I, J, V, N, nb)

    return B
end

"""
    b = build_b_inversion(mesh::FEData, params::Parameters, forcings::Forcings)

Assemble the RHS vector for the inversion problem.
"""
function build_b_inversion(fe_data::FEData, params::Parameters, forcings::Forcings)
    # unpack
    U_test = fe_data.spaces.X_test[1]
    V_test = fe_data.spaces.X_test[2]
    W_test = fe_data.spaces.X_test[3]
    b_diri = fe_data.spaces.b_diri
    dΓ = fe_data.mesh.dΓ
    dΩ = fe_data.mesh.dΩ
    α = params.α
    τˣ = forcings.τˣ
    τʸ = forcings.τʸ

    # allocate vector of length N
    nu, nv, nw, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + nv + nw + np
    b = zeros(N)

    # linear forms
    lx(vx) = ∫( α*vx*τˣ )dΓ
    ly(vy) = ∫( α*vy*τʸ )dΓ
    lz(vz) = ∫( 1/α*(b_diri*vz) )dΩ # correction due to Dirichlet boundary condition

    # assemble
    b[1:nu] .= assemble_vector(lx, U_test) 
    b[nu+1:nu+nv] .= assemble_vector(ly, V_test)
    b[nu+nv+1:nu+nv+nw] .= assemble_vector(lz, W_test)

    return b
end


"""
    A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = 
        build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings; filename="")

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
function build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings; filename="")
    isfile(filename) && @warn "Evolution system file already exists and will be overwritten." filename

    A_adv = build_advection_matrix(fe_data)
    A_hdiff, B_hdiff, b_hdiff = build_hdiffusion_system(fe_data, params, forcings.κₕ)
    A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(fe_data, params, forcings.κᵥ)

    if filename != ""
        jldsave(filename; A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff, params, forcings)
        @info @sprintf("Evolution system saved to '%s' (%.3f GB)", filename, filesize(filename)/1e9)
    end

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
    A, B, b = build_hdiffusion_system(fe_data::FEData, params::Parameters, κₕ)

Assemble the matrices for the horizontal diffusion part of the evolution problem.

See also [`build_diffusion_system`](@ref).
"""
function build_hdiffusion_system(fe_data::FEData, params::Parameters, κₕ)
    return build_diffusion_system(fe_data, params, κₕ, :horizontal)
end

"""
    A, B, b = build_vdiffusion_system(fe_data::FEData, params::Parameters, κᵥ)

Assemble the matrices for the vertical diffusion part of the evolution problem.

See also [`build_diffusion_system`](@ref).
"""
function build_vdiffusion_system(fe_data::FEData, params::Parameters, κᵥ)
    return build_diffusion_system(fe_data, params, κᵥ, :vertical)
end

"""
    A, B, b = build_diffusion_system(fe_data::FEData, params::Parameters, κ, direction::Symbol)

Assemble the matrices for the diffusion part of the evolution problem.

We use the Crank-Nicolson scheme, i.e., 
```math
A b^{n+1} = B b^n + b
```
where ``A = M + θ K`` and ``B = M - θ K`` with ``θ = Δt/4 α² ε² / μϱ`` and `M` and `K` being the mass 
and stiffness matrices, respectively.

`direction` must be either `:horizontal` or `:vertical`.
"""
function build_diffusion_system(fe_data::FEData, params::Parameters, κ, direction::Symbol)
    if direction != :horizontal && direction != :vertical
        throw(ArgumentError("direction must be :horizontal or :vertical"))
    end

    B_trial = fe_data.spaces.B_trial
    B_test = fe_data.spaces.B_test
    dΩ = fe_data.mesh.dΩ
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    b_diri = fe_data.spaces.b_diri

    # coefficient for diffusion step (Δt/2 for Crank-Nicolson and Δt/2 for Strange splitting makes Δt/4)
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

    A = assemble_matrix(a_lhs, B_trial, B_test)
    B = assemble_matrix(a_rhs, B_trial, B_test)
    b   = assemble_vector(d -> a_rhs(b_diri, d), B_test)
    b .-= assemble_vector(d -> a_lhs(b_diri, d), B_test)

    if direction == :vertical
        # vector for nonzero N² (no Δt/2 for Crank-Nicolson here since it's fully on the RHS)
        N² = params.N²
        l(d) = ∫( -2*θ*N²*(κ*∂z(d)) )dΩ
        b .+= assemble_vector(l, B_test)
    end

    return A, B, b
end

"""
    A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = load_evolution_system(params, filename)

Load the matrices for the evolution problem from a file.
"""
function load_evolution_system(params, filename)
    file = jldopen(filename, "r")
    A_adv = file["A_adv"]
    A_hdiff = file["A_hdiff"]
    B_hdiff = file["B_hdiff"]
    b_hdiff = file["b_hdiff"]
    A_vdiff = file["A_vdiff"]
    B_vdiff = file["B_vdiff"]
    b_vdiff = file["b_vdiff"]
    p0 = file["params"]
    close(file)
    params != p0 && @warn "Parameters mismatch detected!" #TODO: also detect κ mismatch
    @info @sprintf("Evolution system loaded from '%s' (%.3f GB)", filename, filesize(filename)/1e9)
    return A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff
end