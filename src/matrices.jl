# gradients 
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

# function build_matrices(mesh::Mesh)
#     A_inversion = build_A_inversion(mesh)
#     B_inversion = build_B_inversion(mesh)
#     A_adv, A_diff = build_A_adv_A_diff(mesh)
#     B_diff, b_diff = build_B_diff(mesh)
#     return A_inversion, B_inversion, A_adv, A_diff, B_diff, b_diff
# end

"""
    A = build_A_inversion(mesh::Mesh, γ, ε², ν, f; fname)

Assemble the LHS matrix `A` for the inversion problem. 
If `fname` is given, the data is saved to a file.
"""
function build_A_inversion(mesh::Mesh, γ, ε², ν, f; fname=nothing)
    # unpack
    X_trial, X_test, dΩ = mesh.X_trial, mesh.X_test, mesh.dΩ

    # bilinear form
    a((ux, uy, uz, p), (vx, vy, vz, q)) =
        ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
           γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
         γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                                                      ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ

    # assemble 
    @time "build A_inversion" A = assemble_matrix(a, X_trial, X_test)

    # save
    if fname !== nothing
        jldsave(fname; A_inversion=A)
        @info @sprintf("A_inversion saved to '%s' (%.1f GB)", fname, filesize(fname)/1e9)
    end

    return A
end

"""
    B = build_B_inversion(mesh::Mesh)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(mesh::Mesh)
    # unpack
    X_test, B_trial, dΩ = mesh.X_test, mesh.B_trial, mesh.dΩ

    # bilinear form
    a(b, vz) = ∫( b*vz )dΩ

    # unpack X_test
    U_test, V_test, W_test, P_test = unpack_spaces(X_test)

    # assemble
    @time "B_inversion" B = assemble_matrix(a, B_trial, W_test)

    # convert to N × nb matrix
    nu, nv, nw, np, nb = get_n_dof(mesh)
    N = nu + nv + nw + np
    I, J, V = findnz(B)
    I .+= nu + nv
    B = sparse(I, J, V, N, nb)

    return B
end

"""
    LHS_adv, LHS_diff, perm, inv_perm = assemble_LHS_adv_diff(arch::AbstractArchitecture, 
                    α, γ, κ, B, D, dΩ; fname_adv="LHS_adv.h5", fname_diff="LHS_diff.h5")

Assemble the LHSs for the advection and diffusion components of the evolution
problem for the PG equations. Return the sparse matrices `LHS_adv` and
`LHS_diff` along with the permutation `perm` and its inverse `inv_perm`. Save
the matrices to separate files `fname_adv` and `fname_diff`.
"""
function assemble_LHS_adv_diff(arch::AbstractArchitecture, α, γ, κ, B, D, dΩ; fname_adv="LHS_adv.h5", fname_diff="LHS_diff.h5")
    # advection matrix
    a_adv(b, d) = ∫( b*d )dΩ
    @time "assemble LHS_adv" LHS_adv = assemble_matrix(a_adv, B, D)

    # diffusion matrix
    a_diff(b, d) = ∫( b*d + α*γ*∂x(b)*∂x(d)*κ + α*γ*∂y(b)*∂y(d)*κ + α*∂z(b)*∂z(d)*κ )dΩ
    @time "assemble LHS_diff" LHS_diff = assemble_matrix(a_diff, B, D)

    # Cuthill-McKee DOF reordering
    @time "RCM perm" perm, inv_perm = RCM_perm(arch, B, D, dΩ)

    # re-order DOFs
    LHS_adv = LHS_adv[perm, perm]
    LHS_diff = LHS_diff[perm, perm]

    # save
    write_sparse_matrix(LHS_adv,  perm, inv_perm; fname=fname_adv)
    write_sparse_matrix(LHS_diff, perm, inv_perm; fname=fname_diff)

    return LHS_adv, LHS_diff, perm, inv_perm
end

"""
    M, v = assemble_RHS_diff(perm, α, γ, κ, N², B, D, dΩ)

Assemble the RHS matrix and vector for the diffusion part of the evolution
problem for the PG equations.
"""
function assemble_RHS_diff(perm, α, γ, κ, N², B, D, dΩ)
    # matrix
    a(b, d) = ∫( b*d - α*γ*∂x(b)*∂x(d)*κ - α*γ*∂y(b)*∂y(d)*κ - α*∂z(b)*∂z(d)*κ )dΩ
    @time "RHS_diff matrix" M = assemble_matrix(a, B, D)[perm, :]

    # vector
    l(d) = ∫( -2*α*∂z(d)*N²*κ )dΩ
    @time "RHS_diff vector" v = assemble_vector(l, D)[perm]

    return M, v
end