struct Mesh{M, X, Y, B, C, O, DO}
    model::M # UnstructuredDiscreteModel
    X_trial::X # trial MultiFieldFESpace for [u, v, w, p]
    X_test::Y # test MultiFieldFESpace for [u, v, w, p]
    B_trial::B # trial SingleFieldFESpace for buoyancy
    B_test::C # test SingleFieldFESpace for buoyancy
    Ω::O # Triangulation
    dΩ::DO # Measure
end

"""
    m = Mesh(fname::AbstractString)
    m = Mesh(model::Gridap.Geometry.UnstructuredDiscreteModel)

Returns a struct holding mesh-related data.
"""
function Mesh(fname::AbstractString)
    model = GmshDiscreteModel(fname)
    return Mesh(model)
end
function Mesh(model::Gridap.Geometry.UnstructuredDiscreteModel)
    X_trial, X_test, B_trial, B_test = setup_FESpaces(model)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)
    return Mesh(model, X_trial, X_test, B_trial, B_test, Ω, dΩ)
end

### functions for working with degrees of freedom

"""
    nu, nv, nw, np, nb = get_n_dof(mesh::Mesh)
    nu, nv, nw, np, nb = get_n_dof(X_trial, B_trial)

Return the number of degrees of freedom for the velocity, pressure, and 
buoyancy fields.
"""
function get_n_dof(mesh::Mesh)
    return get_n_dof(mesh.X_trial, mesh.B_trial)
end
function get_n_dof(X_trial, B_trial)
    U, V, W, P = unpack_spaces(X_trial)
    nu = U.space.nfree
    nv = V.space.nfree
    nw = W.space.nfree
    np = P.space.space.nfree - 1 # zero mean constraint removes one dof
    nb = B_trial.space.nfree
    return nu, nv, nw, np, nb
end

"""
    p_u, p_v, p_w, p_p, p_b = compute_dof_perms(mesh::Mesh)

Compute optimal permutations of the degrees of freedom for each field.
"""
function compute_dof_perms(mesh::Mesh)
    # unpack 
    X_trial, X_test, B_trial, B_test = mesh.X_trial, mesh.X_test, mesh.B_trial, mesh.B_test
    dΩ = mesh.dΩ

    # unpack spaces
    U_trial, V_trial, W_trial, P_trial = unpack_spaces(X_trial)
    U_test, V_test, W_test, P_test = unpack_spaces(X_test)

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
# function compute_dof_perm(arch::GPU, M)
#     return CUSOLVER.symrcm(M) .+ 1
# end

### some utility functions for working with meshes

"""
    p, t = get_p_t(model::Gridap.Geometry.UnstructuredDiscreteModel)
    p, t = get_p_t(fname::AbstractString)

Return the node coordinates `p` and the connectivities `t` of a mesh.
"""
function get_p_t(model::Gridap.Geometry.UnstructuredDiscreteModel)
    # unpack node coords
    nc = model.grid.node_coordinates
    np = length(nc)
    d = length(nc[1])
    p = [nc[i][j] for i ∈ 1:np, j ∈ 1:d]

    # unpack connectivities
    cni = model.grid.cell_node_ids
    nt = length(cni)
    nn = length(cni[1])
    t = [cni[i][j] for i ∈ 1:nt, j ∈ 1:nn]

    return p, t
end
function get_p_t(fname::AbstractString)
    model = GmshDiscreteModel(fname)
    return get_p_t(model)
end

"""
    p_to_t = get_p_to_t(t, np)

Returns a vector of vectors of vectors `p_to_t` such that p_to_t[i] lists
all the [k, j] pairs in `t` that point to the ith node of the mesh of size `np`.
"""
function get_p_to_t(t, np)
    p_to_t = [[] for i ∈ 1:np]
    for k ∈ axes(t, 1)
        for i ∈ axes(t, 2)
            push!(p_to_t[t[k, i]], [k, i])
        end
    end
    return p_to_t
end