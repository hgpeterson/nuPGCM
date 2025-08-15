# gradients 
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

"""
    A_inversion, B_inversion, A_adv, A_diff, B_diff, b_diff =
        build_matrices(fed::FEData, params::Parameters, f, ν, κ, τx, τy; 
                       A_inversion_ofile=nothing,
                       A_adv_ofile=nothing, A_diff_ofile=nothing)

Build the matrices for the inversion and evolution problems of the PG equations.
The matrices are assembled using the finite element method and can be saved to files
if `ofile`s are provided. The matrices are:
- `A_inversion`: LHS matrix for the inversion problem
- `B_inversion`: RHS matrix for the inversion problem
- `A_adv`: LHS matrix for the advection part of the evolution problem
- `A_diff`: LHS matrix for the diffusion part of the evolution problem
- `B_diff`: RHS matrix for the diffusion part of the evolution problem
- `b_diff`: RHS vector for the diffusion part of the evolution problem
The functions `f`, `ν`, and `κ` are the Coriolis parameter, turbulent viscosity, and turbulent 
diffusivity, respectively. 
`τx` and `τy` are the surface stress components in the x and y directions.
"""
function build_matrices(fed::FEData, params::Parameters, f, ν, κ, τx, τy; 
                        A_inversion_ofile=nothing,
                        A_adv_ofile=nothing, A_diff_ofile=nothing)
    A_inversion, B_inversion, b_inversion = build_inversion_matrices(fed, params, f, ν, τx, τy; 
                                   A_inversion_ofile=A_inversion_ofile)
    A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(fed, params, κ;
                                      A_adv_ofile=A_adv_ofile, A_diff_ofile=A_diff_ofile)
    return A_inversion, B_inversion, b_inversion, A_adv, A_diff, B_diff, b_diff
end

"""
    A_inversion, B_inversion = build_inversion_matrices(fed::FEData, params::Parameters, f, ν, τx, τy; 
                                   A_inversion_ofile=nothing)

Build the matrices for the inversion problem of the PG equations.
The matrices are assembled using the finite element method and can be saved to files
if `A_inversion_ofile` is provided. The matrices are:
- `A_inversion`: LHS matrix for the inversion problem
- `B_inversion`: RHS matrix for the inversion problem
- `b_inversion`: RHS vectory for the inversion problem
The functions `f` and `ν` are the Coriolis parameter and turbulent viscosity, respectively.
`τx` and `τy` are the surface stress components in the x and y directions.
"""
function build_inversion_matrices(fed::FEData, params::Parameters, f, ν, τx, τy; 
                                  A_inversion_ofile=nothing)
    A_inversion = build_A_inversion(fed, params, f, ν; ofile=A_inversion_ofile)
    B_inversion = build_B_inversion(fed, params)
    b_inversion = build_b_inversion(fed, params, τx, τy)
    return A_inversion, B_inversion, b_inversion
end

"""
    A = build_A_inversion(fed::FEData, params::Parameters, f, ν; ofile)

Assemble the LHS matrix `A` for the inversion problem. 
If `ofile` is given, the data is saved to a file.
"""
function build_A_inversion(fed::FEData, params::Parameters, f, ν; ofile=nothing)
    # unpack
    X_trial = fed.spaces.X_trial
    X_test = fed.spaces.X_test
    dΩ = fed.mesh.dΩ
    ε = params.ε
    α = params.α

    # coefficient
    α²ε² = α^2*ε^2

    # bilinear form !! ASSUMES ν IS A CONSTANT !!
    a((ux, uy, uz, p), (vx, vy, vz, q)) =
        ∫(α²ε²*∇(ux)⋅∇(vx)*ν - uy*vx*f + ∂x(p)*vx +
          α²ε²*∇(uy)⋅∇(vy)*ν + ux*vy*f + ∂y(p)*vy +
          α²ε²*∇(uz)⋅∇(vz)*ν +           ∂z(p)*vz +
                   ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ

    # assemble 
    @time "build A_inversion" A = assemble_matrix(a, X_trial, X_test)

    # save
    if ofile !== nothing
        jldsave(ofile; A_inversion=A, params=params, f=f, ν=ν)
        @info @sprintf("A_inversion saved to '%s' (%.3f GB)", ofile, filesize(ofile)/1e9)
    end

    return A
end

"""
    B = build_B_inversion(fed::FEData, params)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(fed::FEData, params::Parameters)
    # unpack
    W_test = fed.spaces.X_test[3]
    B_trial = fed.spaces.B_trial
    dΩ = fed.mesh.dΩ
    α = params.α

    # coefficient
    α⁻¹ = 1/α

    # bilinear form
    a(b, vz) = ∫( α⁻¹*b*vz )dΩ

    # assemble
    B = assemble_matrix(a, B_trial, W_test)

    # convert to N × nb matrix
    nu, nv, nw, np, nb = get_n_dofs(fed.dofs)
    N = nu + nv + nw + np
    I, J, V = findnz(B)
    I .+= nu + nv
    B = sparse(I, J, V, N, nb)

    return B
end

"""
    b = build_b_inversion(mesh::FEData, params::Parameters, τx, τy)

Assemble the RHS vector for the inversion problem.
"""
function build_b_inversion(fed::FEData, params::Parameters, τx, τy)
    # unpack
    U_test = fed.spaces.X_test[1]
    V_test = fed.spaces.X_test[2]
    dΓ = fed.mesh.dΓ
    α = params.α

    # allocate vector of length N
    nu, nv, nw, np, nb = get_n_dofs(fed.dofs)
    N = nu + nv + nw + np
    b = zeros(N)

    # linear forms
    lx(vx) = ∫( α*vx*τx )dΓ
    ly(vy) = ∫( α*vy*τy )dΓ

    # assemble
    b[1:nu] .= assemble_vector(lx, U_test) 
    b[nu+1:nu+nv] .= assemble_vector(ly, V_test)

    return b
end


"""
    A_adv, b_adv, A_diff, B_diff, b_diff = build_evolution_system(fed::FEData, params::Parameters, κ; 
                        filename="", force_build=false)

Assemble or load matrices and vectors for the PG evolution equation.
"""
function build_evolution_system(fed::FEData, params::Parameters, κ; filename="", force_build=false)
    if !isfile(filename) || force_build
        !isfile(filename) && @warn "Evolution system file not found, building..." filename
        force_build && @warn "`force_build` set to `true`, building evolution system..." filename

        # unpack
        B_trial = fed.spaces.B_trial
        B_test = fed.spaces.B_test
        dΩ = fed.mesh.dΩ
        ε = params.ε
        α = params.α
        μϱ = params.μϱ
        Δt = params.Δt
        N² = params.N²

        # coefficient for diffusion step (Δt/2 for Crank-Nicolson and Δt/2 for Strange splitting makes Δt/4)
        θ = Δt/4 * α^2 * ε^2 / μϱ

        # bilinear forms for advection and diffusion
        a_adv(b, d) = ∫( b*d )dΩ
        a_diff_lhs(b, d) = ∫( b*d + θ*(κ*∇(b)⋅∇(d)) )dΩ
        a_diff_rhs(b, d) = ∫( b*d - θ*(κ*∇(b)⋅∇(d)) )dΩ

        # assemble matrices
        A_adv = assemble_matrix(a_adv, B_trial, B_test)
        A_diff = assemble_matrix(a_diff_lhs, B_trial, B_test)
        B_diff = assemble_matrix(a_diff_rhs, B_trial, B_test)

        # assemble vectors (b_diri = b_diri_rhs - b_diri_lhs)
        b_adv = -build_diri_vector(a_adv, fed.spaces.b_diri, B_test)
        b_diff = build_diri_vector(a_diff_rhs, fed.spaces.b_diri, B_test)
        b_diff .-= build_diri_vector(a_diff_lhs, fed.spaces.b_diri, B_test)

        # vector for nonzero N² (no Δt/2 for Crank-Nicolson here since it's fully on the RHS)
        l(d) = ∫( -2*θ*N²*(κ*∂z(d)) )dΩ
        b_diff .+= assemble_vector(l, B_test)

        jldsave(filename; A_adv, b_adv, A_diff, B_diff, b_diff, params, κ)
        @info @sprintf("Evolution system saved to '%s' (%.3f GB)", filename, filesize(filename)/1e9)
    else
        file = jldopen(filename, "r")
        A_adv = file["A_adv"]
        b_adv = file["b_adv"]
        A_diff = file["A_diff"]
        B_diff = file["B_diff"]
        b_diff = file["b_diff"]
        p0 = file["params"]
        κ0 = file["κ"]
        close(file)
        params != p0 && @warn "Parameters mismatch detected!"
        κ != κ0 && @warn "Turbulent diffusivity mismatch detected!"
        @info @sprintf("Evolution system loaded from '%s' (%.3f GB)", filename, filesize(filename)/1e9)
    end

    return A_adv, b_adv, A_diff, B_diff, b_diff
end
function build_diri_vector(a, b_diri, B_test)
    return assemble_vector(d -> a(b_diri, d), B_test)
end