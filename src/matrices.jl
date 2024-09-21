# gradients 
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

"""
    LHS, perm, inv_perm = assemble_LHS_inversion(arch::AbstractArchitecture, γ, ε², ν, f, X, Y, dΩ; fname="LHS_inversion.h5")

Assemble the LHS of the inversion problem for the Non-Hydrostatic PG equations 
and return the matrix `LHS` along with the permutation `perm` and its
inverse `inv_perm`. The matrix is saved to a file `fname`.
"""
function assemble_LHS_inversion(arch::AbstractArchitecture, γ, ε², ν, f, X, Y, dΩ; fname="LHS_inversion.h5")
    # bilinear form
    a((ux, uy, uz, p), (vx, vy, vz, q)) = 
        ∫(   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
             ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
           γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
        # ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
        #    γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
        #  γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                                                      ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ

    # assemble 
    @time "assemble LHS_inversion" LHS = assemble_matrix(a, X, Y)

    # Cuthill-McKee DOF reordering
    @time "RCM perm" perm, inv_perm = RCM_perm(arch, X, Y, dΩ)

    # re-order DOFs
    LHS = LHS[perm, perm]

    # save
    write_sparse_matrix(LHS, perm, inv_perm; fname)

    return LHS, perm, inv_perm
end

"""
    RHS = assemble_RHS_inversion(perm_inversion, B::TrialFESpace, Y::MultiFieldFESpace, dΩ::Measure)

Assemble the RHS matrix for the inversion problem of the Non-Hydrostatic PG equations.
"""
function assemble_RHS_inversion(perm_inversion, B::TrialFESpace, Y::MultiFieldFESpace, dΩ::Measure)
    # bilinear form
    a(b, vz) = ∫( b*vz )dΩ

    # unpack Y
    Vx, Vy, Vz, Q = unpack_spaces(Y)
    nx = Vx.nfree
    ny = Vy.nfree
    nz = Vz.nfree

    # permutation for Uz
    perm_uz = perm_inversion[nx+ny+1:nx+ny+nz] .- nx .- ny

    # assemble
    @time "RHS_inversion" RHS = assemble_matrix(a, B, Vz)[perm_uz, :]

    return RHS
end

"""
    LHS, perm, inv_perm = assemble_LHS_evolution(arch::AbstractArchitecture, α, γ, κ, B, D, dΩ; fname="LHS_evolution.h5")

Assemble the LHS of the evolution problem for the Non-Hydrostatic PG equations
and return the matrix `LHS` along with the permutation `perm` and its
inverse `inv_perm`. The matrix is saved to a file `fname`.
"""
function assemble_LHS_evolution(arch::AbstractArchitecture, α, γ, κ, B, D, dΩ; fname="LHS_evolution.h5")
    # bilinear form
    a(b, d) = ∫( b*d + α*∂z(b)*∂z(d)*κ )dΩ
    # a(b, d) = ∫( b*d + α*γ*∂x(b)*∂x(d)*κ + α*γ*∂y(b)*∂y(d)*κ + α*∂z(b)*∂z(d)*κ )dΩ

    # assemble
    @time "assemble LHS_evolution" LHS = assemble_matrix(a, B, D)

    # Cuthill-McKee DOF reordering
    @time "RCM perm" perm, inv_perm = RCM_perm(arch, B, D, dΩ)

    # re-order DOFs
    LHS = LHS[perm, perm]

    # save
    write_sparse_matrix(LHS, perm, inv_perm; fname)

    return LHS, perm, inv_perm
end

"""
    perm, inv_perm = RCM_perm(arch::AbstractArchitecture, X::MultiFieldFESpace, 
                              Y::MultiFieldFESpace, dΩ::Measure)

Return the reverse Cuthill-McKee permutation and its inverse for a multi-field FE 
space.
"""
function RCM_perm(arch::AbstractArchitecture, X::MultiFieldFESpace, Y::MultiFieldFESpace, dΩ::Measure)
    # unpack spaces
    Ux, Uy, Uz, P = unpack_spaces(X)
    Vx, Vy, Vz, Q = unpack_spaces(Y)
    nx = Ux.space.nfree
    ny = Uy.space.nfree
    nz = Uz.space.nfree

    # assemble mass matrices
    a(u, v) = ∫( u*v )dΩ
    M_ux = assemble_matrix(a, Ux, Vx)
    M_uy = assemble_matrix(a, Uy, Vy)
    M_uz = assemble_matrix(a, Uz, Vz)
    M_p  = assemble_matrix(a, P, Q)

    # compute permutations
    perm_ux = RCM_perm(arch, M_ux)
    perm_uy = RCM_perm(arch, M_uy)
    perm_uz = RCM_perm(arch, M_uz)
    perm_p  = RCM_perm(arch, M_p)

    # combine
    perm = [perm_ux; 
            perm_uy .+ nx; 
            perm_uz .+ nx .+ ny; 
            perm_p  .+ nx .+ ny .+ nz]

    # inverse permutation
    inv_perm = invperm(perm)

    return perm, inv_perm
end

"""
    perm, inv_perm = RCM_perm(arch::AbstractArchitecture, B::Gridap.FESpaces.SingleFieldFESpace, 
                              D::Gridap.FESpaces.SingleFieldFESpace, dΩ::Measure)

Return the reverse Cuthill-McKee permutation and its inverse for a single-field FE 
space.
"""
function RCM_perm(arch::AbstractArchitecture, B::Gridap.FESpaces.SingleFieldFESpace, 
                  D::Gridap.FESpaces.SingleFieldFESpace, dΩ::Measure)
    # assemble mass matrix
    a(b, d) = ∫( b*d )dΩ
    M_b = assemble_matrix(a, B, D)

    # compute permutation
    perm = RCM_perm(arch, M_b)

    # inverse permutation
    inv_perm = invperm(perm)

    return perm, inv_perm
end

"""
    perm = RCM_perm(arch::AbstractArchitecture, M)

Return the reverse Cuthill-McKee permutation of the matrix `M`. If `arch` is 
`GPU`, the permutation is computed using the CUSOLVER library, otherwise it 
is computed using the CuthillMcKee library.
"""
function RCM_perm(arch::GPU, M)
    return CUSOLVER.symrcm(M) .+ 1
end
function RCM_perm(arch::CPU, M)
    return CuthillMcKee.symrcm(M, true, false)
end