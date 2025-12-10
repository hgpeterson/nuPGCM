struct DoFHandler{P, I}
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
    nu::I     # number of dofs for each field
    nv::I
    nw::I
    np::I
    nb::I
end

function Base.summary(dofs::DoFHandler)
    t = typeof(dofs)
    return "$t with (nu=$(dofs.nu), nv=$(dofs.nv), nw=$(dofs.nw), np=$(dofs.np), nb=$(dofs.nb)) DOFs"
end
function Base.show(io::IO, dofs::DoFHandler)
    println(io, summary(dofs), ":")
    println(io, "├── p_u and inv_p_u: ", summary(dofs.p_u), "'s")
    println(io, "├── p_v and inv_p_v: ", summary(dofs.p_v), "'s")
    println(io, "├── p_w and inv_p_w: ", summary(dofs.p_w), "'s")
    println(io, "├── p_p and inv_p_p: ", summary(dofs.p_p), "'s")
    println(io, "├── p_inversion and inv_p_inversion: ", summary(dofs.p_inversion), "'s")
      print(io, "└── p_b and inv_p_b: ", summary(dofs.p_b), "'s")
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
    U, UB, V, VB, W, WB, P = X_trial
    nu = U.space.nfree + UB.space.nfree
    nv = V.space.nfree + VB.space.nfree
    nw = W.space.nfree + WB.space.nfree
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
    U_trial, UB_trial, V_trial, VB_trial, W_trial, WB_trial, P_trial = X_trial
    U_test, UB_test, V_test, VB_test, W_test, WB_test, P_test = X_test
    U_mini_trial = MultiFieldFESpace([U_trial, UB_trial])
    U_mini_test  = MultiFieldFESpace([U_test, UB_test])
    V_mini_trial = MultiFieldFESpace([V_trial, VB_trial])
    V_mini_test  = MultiFieldFESpace([V_test, VB_test])
    W_mini_trial = MultiFieldFESpace([W_trial, WB_trial])
    W_mini_test  = MultiFieldFESpace([W_test, WB_test])

    # assemble mass matrices for each field
    a(u, v) = ∫( u*v )dΩ
    a_mini((u,ub), (v,vb)) = a(u+ub, v+vb)
    M_u = assemble_matrix(a_mini, U_mini_trial, U_mini_test)
    M_v = assemble_matrix(a_mini, V_mini_trial, V_mini_test)
    M_w = assemble_matrix(a_mini, W_mini_trial, W_mini_test)
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

struct FEData{M<:Mesh, S<:Spaces, D<:DoFHandler}
    mesh::M    # mesh data
    spaces::S  # finite element spaces
    dofs::D    # degrees of freedom handler
end

function Base.summary(fe_data::FEData)
    t = typeof(fe_data)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, fe_data::FEData)
    println(io, summary(fe_data), ":")
    println(io, "├── mesh: ", summary(fe_data.mesh))
    println(io, "├── spaces: ", summary(fe_data.spaces))
      print(io, "└── dofs: ", summary(fe_data.dofs))
end

function FEData(mesh::Mesh, spaces::Spaces)
    dofs = DoFHandler(spaces, mesh.dΩ)
    return FEData(mesh, spaces, dofs)
end