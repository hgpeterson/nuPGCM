struct DoFHandler{P, I}
    p_u::P      # dof permutations
    p_p::P
    p_inversion::P
    p_b::P
    inv_p_u::P  # inverse dof permutations
    inv_p_p::P
    inv_p_inversion::P
    inv_p_b::P
    nu::I     # number of dofs for each field
    np::I
    nb::I
end

function Base.summary(dofs::DoFHandler)
    t = typeof(dofs)
    return "$t with (nu=$(dofs.nu), np=$(dofs.np), nb=$(dofs.nb)) DOFs"
end
function Base.show(io::IO, dofs::DoFHandler)
    println(io, summary(dofs), ":")
    println(io, "├── p_u and inv_p_u: ", summary(dofs.p_u), "'s")
    println(io, "├── p_p and inv_p_p: ", summary(dofs.p_p), "'s")
    println(io, "├── p_inversion and inv_p_inversion: ", summary(dofs.p_inversion), "'s")
      print(io, "└── p_b and inv_p_b: ", summary(dofs.p_b), "'s")
end

function DoFHandler(spaces, dΩ)
    p_u, p_p, p_b = compute_dof_perms(spaces, dΩ)
    return DoFHandler(p_u, p_p, p_b)
end
function DoFHandler(p_u, p_p, p_b)
    nu = length(p_u)
    np = length(p_p)
    nb = length(p_b)
    inv_p_u = invperm(p_u)
    inv_p_p = invperm(p_p)
    inv_p_b = invperm(p_b)
    p_inversion = [p_u; nu.+ p_p]
    inv_p_inversion = invperm(p_inversion)
    return DoFHandler(p_u, p_p, p_inversion, p_b, inv_p_u, inv_p_p, inv_p_inversion, inv_p_b, nu, np, nb)
end

"""
    nu, np, nb = get_n_dofs(spaces::Spaces)
    nu, np, nb = get_n_dofs(X_trial, B_trial)
    nu, np, nb = get_n_dofs(dof::DoFHandler)

Return the number of degrees of freedom for the velocity, pressure, and 
buoyancy fields.
"""
function get_n_dofs(spaces::Spaces)
    return get_n_dofs(spaces.X_trial, spaces.B_trial)
end
function get_n_dofs(X_trial, B_trial)
    U, P = X_trial[1], X_trial[2]
    nu = U.space.nfree
    np = P.space.space.nfree - 1 # zero mean constraint removes one dof
    nb = B_trial.space.nfree
    return nu, np, nb
end
function get_n_dofs(dofs::DoFHandler)
    return dofs.nu, dofs.np, dofs.nb
end

"""
    p_u, p_p, p_b = compute_dof_perms(spaces::Spaces, dΩ)

Compute optimal permutations of the degrees of freedom for each field.
"""
function compute_dof_perms(spaces::Spaces, dΩ)
    # unpack 
    U_trial = spaces.X_trial[1]
    P_trial = spaces.X_trial[2]
    B_trial = spaces.B_trial
    U_test = spaces.X_test[1]
    P_test = spaces.X_test[2]
    B_test = spaces.B_test

    # assemble mass matrices for each field
    a(u, v) = ∫( u⋅v )dΩ
    M_u = assemble_matrix(a, U_trial, U_test)
    M_p = assemble_matrix(a, P_trial, P_test)
    M_b = assemble_matrix(a, B_trial, B_test)

    # compute permutations
    p_u = compute_dof_perm(M_u)
    p_p = compute_dof_perm(M_p)
    p_b = compute_dof_perm(M_b)

    return p_u, p_p, p_b
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