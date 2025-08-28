struct DoFHandler{P}
    p_u::P      # dof permutations
    p_v::P
    p_w::P
    p_p::P
    p_inversion::P
    p_b::P
    inv_p_u::P  # inverse dof permutations
    inv_p_v::P
    inv_p_w::P
    inv_p_p::P
    inv_p_inversion::P
    inv_p_b::P
    nu::Int     # number of dofs for each field
    nv::Int
    nw::Int
    np::Int
    nb::Int
end

function DoFHandler(spaces, dΩ)
    p_u, p_v, p_w, p_p, p_b = compute_dof_perms(spaces, dΩ)
    return DoFHandler(p_u, p_v, p_w, p_p, p_b)
end
function DoFHandler(p_u, p_v, p_w, p_p, p_b)
    nu = length(p_u)
    nv = length(p_v)
    nw = length(p_w)
    np = length(p_p)
    nb = length(p_b)
    inv_p_u = invperm(p_u)
    inv_p_v = invperm(p_v)
    inv_p_w = invperm(p_w)
    inv_p_p = invperm(p_p)
    inv_p_b = invperm(p_b)
    p_inversion = [p_u; p_v .+ nu; p_w .+ nu .+ nv; p_p .+ nu .+ nv .+ nw]
    inv_p_inversion = invperm(p_inversion)
    return DoFHandler(p_u, p_v, p_w, p_p, p_inversion, p_b, inv_p_u, inv_p_v, inv_p_w, inv_p_p, inv_p_inversion, inv_p_b, nu, nv, nw, np, nb)
end

"""
    nu, nv, nw, np, nb = get_n_dofs(spaces::Spaces)
    nu, nv, nw, np, nb = get_n_dofs(X_trial, B_trial)
    nu, nv, nw, np, nb = get_n_dofs(dof::DoFHandler)

Return the number of degrees of freedom for the velocity, pressure, and 
buoyancy fields.
"""
function get_n_dofs(spaces::Spaces)
    return get_n_dofs(spaces.X_trial, spaces.B_trial)
end
function get_n_dofs(X_trial, B_trial)
    U, V, W, P = X_trial[1], X_trial[2], X_trial[3], X_trial[4]
    nu = U.space.nfree
    nv = V.space.nfree
    nw = W.space.nfree
    np = P.space.space.nfree - 1 # zero mean constraint removes one dof
    nb = B_trial.space.nfree
    return nu, nv, nw, np, nb
end
function get_n_dofs(dofs::DoFHandler)
    return dofs.nu, dofs.nv, dofs.nw, dofs.np, dofs.nb
end

"""
    p_u, p_v, p_w, p_p, p_b = compute_dof_perms(spaces::Spaces, dΩ)

Compute optimal permutations of the degrees of freedom for each field.
"""
function compute_dof_perms(spaces::Spaces, dΩ)
    # unpack 
    X_trial, X_test, B_trial, B_test = spaces.X_trial, spaces.X_test, spaces.B_trial, spaces.B_test

    # unpack spaces
    U_trial, V_trial, W_trial, P_trial = X_trial[1], X_trial[2], X_trial[3], X_trial[4]
    U_test, V_test, W_test, P_test = X_test[1], X_test[2], X_test[3], X_test[4]

    # assemble mass matrices for each field
    a(u, v) = ∫( u*v )dΩ
    M_u = assemble_matrix(a, U_trial, U_test)
    M_v = assemble_matrix(a, V_trial, V_test)
    M_w = assemble_matrix(a, W_trial, W_test)
    M_p = assemble_matrix(a, P_trial, P_test)
    M_b = assemble_matrix(a, B_trial, B_test)

    # compute permutations
    p_u = compute_dof_perm(M_u)
    p_v = compute_dof_perm(M_v)
    p_w = compute_dof_perm(M_w)
    p_p = compute_dof_perm(M_p)
    p_b = compute_dof_perm(M_b)

    return p_u, p_v, p_w, p_p, p_b
end

"""
    perm = compute_dof_perm(M)

Compute the Cuthill-McKee degree of freedom permutation for a mass matrix `M`.
"""
function compute_dof_perm(M)
    return CuthillMcKee.symrcm(M, true, false)
end

### struct to hold Finite Element data

struct FEData{M, S, D}
    mesh::M    # mesh data
    spaces::S  # finite element spaces
    dofs::D    # degrees of freedom handler
end

function FEData(mesh::Mesh, spaces::Spaces)
    dofs = DoFHandler(spaces, mesh.dΩ)
    return FEData(mesh, spaces, dofs)
end