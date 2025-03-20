# gradients 
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

"""
    A_inversion, B_inversion, A_adv, A_diff, B_diff, b_diff =
        build_matrices(mesh, params, f, ν, κ; 
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
"""
function build_matrices(mesh::Mesh, params::Parameters, f, ν, κ; 
                        A_inversion_ofile=nothing,
                        A_adv_ofile=nothing, A_diff_ofile=nothing)
    A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; 
                                   A_inversion_ofile=A_inversion_ofile)
    A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ;
                                      A_adv_ofile=A_adv_ofile, A_diff_ofile=A_diff_ofile)
    return A_inversion, B_inversion, A_adv, A_diff, B_diff, b_diff
end

"""
    A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; 
                                   A_inversion_ofile=nothing)

Build the matrices for the inversion problem of the PG equations.
The matrices are assembled using the finite element method and can be saved to files
if `A_inversion_ofile` is provided. The matrices are:
- `A_inversion`: LHS matrix for the inversion problem
- `B_inversion`: RHS matrix for the inversion problem
The functions `f` and `ν` are the Coriolis parameter and turbulent viscosity, respectively.
"""
function build_inversion_matrices(mesh::Mesh, params::Parameters, f, ν; 
                                  A_inversion_ofile=nothing)
    A_inversion = build_A_inversion(mesh, params, f, ν; ofile=A_inversion_ofile)
    B_inversion = build_B_inversion(mesh)
    return A_inversion, B_inversion
end

"""
    A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ; 
                                   A_adv_ofile=nothing, A_diff_ofile=nothing)
Build the matrices for the evolution problem of the PG equations.
The matrices are assembled using the finite element method and can be saved to files
if `A_adv_ofile` and `A_diff_ofile` are provided. The matrices are:
- `A_adv`: LHS matrix for the advection part of the evolution problem
- `A_diff`: LHS matrix for the diffusion part of the evolution problem
- `B_diff`: RHS matrix for the diffusion part of the evolution problem
- `b_diff`: RHS vector for the diffusion part of the evolution problem
The function `κ` is the turbulent diffusivity.
"""
function build_evolution_matrices(mesh::Mesh, params::Parameters, κ; 
                                   A_adv_ofile=nothing, A_diff_ofile=nothing)
    A_adv, A_diff = build_A_adv_A_diff(mesh, params, κ; 
                        ofile_adv=A_adv_ofile, ofile_diff=A_diff_ofile)
    B_diff, b_diff = build_B_diff_b_diff(mesh, params, κ)
    return A_adv, A_diff, B_diff, b_diff
end

"""
    A = build_A_inversion(mesh, params, f, ν; ofile)

Assemble the LHS matrix `A` for the inversion problem. 
If `ofile` is given, the data is saved to a file.
"""
function build_A_inversion(mesh::Mesh, params::Parameters, f, ν; ofile=nothing)
    # unpack
    X_trial = mesh.spaces.X_trial
    X_test = mesh.spaces.X_test
    dΩ = mesh.dΩ
    ε = params.ε
    α = params.α

    # coefficient
    ε² = ε^2
    α²ε² = α^2*ε^2
    α⁴ε² = α^4*ε^2

    # bilinear form
    a((ux, uy, uz, p), (vx, vy, vz, q)) =
        ∫( α²ε²*∂x(ux)*∂x(vx)*ν + α²ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
           α²ε²*∂x(uy)*∂x(vy)*ν + α²ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
           α⁴ε²*∂x(uz)*∂x(vz)*ν + α⁴ε²*∂y(uz)*∂y(vz)*ν + α²ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                                                     ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
    # a((ux, uy, uz, p), (vx, vy, vz, q)) =
    #     ∫(α²ε²*∂x(ux)*∂x(vx)*ν + α²ε²*∂y(ux)*∂y(vx)*ν + α²ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
    #       α²ε²*∂x(uy)*∂x(vy)*ν + α²ε²*∂y(uy)*∂y(vy)*ν + α²ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
    #       α²ε²*∂x(uz)*∂x(vz)*ν + α²ε²*∂y(uz)*∂y(vz)*ν + α²ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
    #                                                                 ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ

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
    B = build_B_inversion(mesh)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(mesh::Mesh)
    # unpack
    W_test = mesh.spaces.X_test[3]
    B_trial = mesh.spaces.B_trial
    dΩ = mesh.dΩ

    # bilinear form
    a(b, vz) = ∫( b*vz )dΩ

    # assemble
    @time "build B_inversion" B = assemble_matrix(a, B_trial, W_test)

    # convert to N × nb matrix
    nu, nv, nw, np, nb = get_n_dofs(mesh.dofs)
    N = nu + nv + nw + np
    I, J, V = findnz(B)
    I .+= nu + nv
    B = sparse(I, J, V, N, nb)

    return B
end
function build_B_inversion(mesh::Mesh, params::Parameters)
    # unpack
    W_test = mesh.spaces.X_test[3]
    B_trial = mesh.spaces.B_trial
    dΩ = mesh.dΩ
    α = params.α

    # coefficient
    α⁻¹ = 1/α

    # bilinear form
    a(b, vz) = ∫( α⁻¹*b*vz )dΩ

    # assemble
    @time "build B_inversion" B = assemble_matrix(a, B_trial, W_test)

    # convert to N × nb matrix
    nu, nv, nw, np, nb = get_n_dofs(mesh.dofs)
    N = nu + nv + nw + np
    I, J, V = findnz(B)
    I .+= nu + nv
    B = sparse(I, J, V, N, nb)

    return B
end

"""
    A_adv, A_diff = build_A_adv_A_diff(mesh, params, κ; ofile_adv, ofile_diff)

Assemble the LHS matrices for the advection and diffusion components of the evolution
problem for the PG equations. If `ofile`s are given, the data is saved to files.
"""
function build_A_adv_A_diff(mesh::Mesh, params::Parameters, κ; ofile_adv=nothing, ofile_diff=nothing)
    # unpack
    B_trial = mesh.spaces.B_trial
    B_test = mesh.spaces.B_test
    dΩ = mesh.dΩ
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt

    # advection matrix
    a_adv(b, d) = ∫( b*d )dΩ
    @time "build A_adv" A_adv = assemble_matrix(a_adv, B_trial, B_test)

    # diffusion matrix
    θ = Δt/2 * ε^2 / μϱ
    θα² = θ*α^2
    a_diff(b, d) = ∫( b*d + θα²*∂x(b)*∂x(d)*κ + θα²*∂y(b)*∂y(d)*κ + θ*∂z(b)*∂z(d)*κ )dΩ
    @time "build A_diff" A_diff = assemble_matrix(a_diff, B_trial, B_test)

    if ofile_adv !== nothing
        jldsave(ofile_adv; A_adv=A_adv, params=params, κ=κ, Δt=Δt)
        @info @sprintf("A_adv saved to '%s' (%.3f GB)", ofile_adv, filesize(ofile_adv)/1e9)
    end
    if ofile_diff !== nothing
        jldsave(ofile_diff; A_diff=A_diff, params=params, κ=κ, Δt=Δt)
        @info @sprintf("A_diff saved to '%s' (%.3f GB)", ofile_diff, filesize(ofile_diff)/1e9)
    end

    return A_adv, A_diff
end

"""
    B, b = build_B_diff_b_diff(mesh, params, κ)

Assemble the RHS matrix and vector for the diffusion part of the evolution
problem for the PG equations.
"""
function build_B_diff_b_diff(mesh::Mesh, params::Parameters, κ)
    # unpack
    B_trial = mesh.spaces.B_trial
    B_test = mesh.spaces.B_test
    dΩ = mesh.dΩ
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    N² = params.N²

    # matrix
    θ = Δt/2 * ε^2 / μϱ
    θα² = θ*α^2
    a(b, d) = ∫( b*d - θα²*∂x(b)*∂x(d)*κ - θα²*∂y(b)*∂y(d)*κ - θ*∂z(b)*∂z(d)*κ )dΩ
    @time "build B_diff" B = assemble_matrix(a, B_trial, B_test)

    # vector
    θN² = θ*N²
    l(d) = ∫( -2θN²*∂z(d)*κ )dΩ
    @time "build b_diff" b = assemble_vector(l, B_test)

    return B, b
end