struct DoFHandler{P, I}
    p_u::P      # dof permutations
    p_ub::P     
    p_p::P
    p_inversion::P
    p_b::P
    inv_p_u::P  # inverse dof permutations
    inv_p_ub::P
    inv_p_p::P
    inv_p_inversion::P
    inv_p_b::P
    nu::I     # number of dofs for each field
    nub::I 
    np::I
    nb::I
end

function Base.summary(dofs::DoFHandler)
    t = typeof(dofs)
    return "$t with (nu=$(dofs.nu), nub=$(dofs.nub), np=$(dofs.np), nb=$(dofs.nb)) DOFs"
end
function Base.show(io::IO, dofs::DoFHandler)
    println(io, summary(dofs), ":")
    println(io, "├── p_u and inv_p_u: ", summary(dofs.p_u), "'s")
    println(io, "├── p_ub and inv_p_ub: ", summary(dofs.p_ub), "'s")
    println(io, "├── p_p and inv_p_p: ", summary(dofs.p_p), "'s")
    println(io, "├── p_inversion and inv_p_inversion: ", summary(dofs.p_inversion), "'s")
      print(io, "└── p_b and inv_p_b: ", summary(dofs.p_b), "'s")
end

function DoFHandler(spaces, dΩ)
    p_u, p_ub, p_p, p_b = compute_dof_perms(spaces, dΩ)
    return DoFHandler(p_u, p_ub, p_p, p_b)
end
function DoFHandler(p_u, p_ub, p_p, p_b)
    nu = length(p_u)
    nub = length(p_ub)
    np = length(p_p)
    nb = length(p_b)
    inv_p_u = invperm(p_u)
    inv_p_ub = invperm(p_ub)
    inv_p_p = invperm(p_p)
    inv_p_b = invperm(p_b)
    p_inversion = [p_u; p_ub .+ nu; p_p .+ nu .+ nub]
    inv_p_inversion = invperm(p_inversion)
    return DoFHandler(p_u, p_ub, p_p, p_inversion, p_b, inv_p_u, inv_p_ub, inv_p_p, inv_p_inversion, inv_p_b, nu, nub, np, nb)
end

"""
    nu, nub, np, nb = get_n_dofs(spaces::Spaces)
    nu, nub, np, nb = get_n_dofs(X, B)
    nu, nub, np, nb = get_n_dofs(dof::DoFHandler)

Return the number of degrees of freedom for the velocity, pressure, and 
buoyancy fields.
"""
function get_n_dofs(spaces::Spaces)
    return get_n_dofs(spaces.X, spaces.B)
end
function get_n_dofs(X, B)
    U, UB, P = X
    nu = U.space.nfree 
    nub = UB.space.nfree
    np = P.space.space.nfree - 1 # zero mean constraint removes one dof
    nb = B.space.nfree
    return nu, nub, np, nb
end
function get_n_dofs(dofs::DoFHandler)
    return dofs.nu, dofs.nub, dofs.np, dofs.nb
end

"""
    p_u, p_ub, p_p, p_b = compute_dof_perms(spaces::Spaces, dΩ)

Compute optimal permutations of the degrees of freedom for each field.
"""
function compute_dof_perms(spaces::Spaces, dΩ)
    # unpack 
    X, Y, B, D = spaces.X, spaces.Y, spaces.B, spaces.D

    # unpack spaces
    U, UB, P = X
    V, VB, Q = Y

    # assemble mass matrices for each field
    a(u, v) = ∫( u⋅v )dΩ
    M_u = assemble_matrix(a, U, V)
    M_ub = assemble_matrix(a, UB, VB)
    M_p = assemble_matrix(a, P, Q)
    M_b = assemble_matrix(a, B, D)

    # compute permutations
    p_u = compute_dof_perm(M_u)
    p_ub = compute_dof_perm(M_ub)
    p_p = compute_dof_perm(M_p)
    p_b = compute_dof_perm(M_b)

    return p_u, p_ub, p_p, p_b
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