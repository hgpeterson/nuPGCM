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

struct FEData{M<:Mesh, S<:Spaces, D<:DoFHandler, N, K, A}
    mesh::M    # mesh data
    spaces::S  # finite element spaces
    dofs::D    # degrees of freedom handler
    ν::N       # actual viscosity FEFunction (enhanced in low-stratification regions)
    ν₀::N      # original viscosity FEFunction
    κᵥ::K      # actual vertical diffusivity FEFunction (enhanced in convective regions)
    κᵥ₀::K     # original vertical diffusivity FEFunction
    Mν::A      # mass matrix for ν
    Mκ::A      # mass matrix for κᵥ
end

function FEData(mesh::Mesh, spaces::Spaces, forcings::Forcings)
    dofs = DoFHandler(spaces, mesh.dΩ)

    # interpolate forcings onto FE spaces so we can update
    ν   = interpolate_everywhere(forcings.ν,  spaces.ν_trial)
    ν₀  = interpolate_everywhere(forcings.ν,  spaces.ν_trial)
    κᵥ  = interpolate_everywhere(forcings.κᵥ, spaces.κ_trial)
    κᵥ₀ = interpolate_everywhere(forcings.κᵥ, spaces.κ_trial)

    # mass matrices for ν and κᵥ
    dΩ = mesh.dΩ
    a(u, v) = ∫( u*v )dΩ
    Mν = assemble_matrix(a, spaces.ν_trial, spaces.ν_test)
    Mν = lu(Mν)
    Mκ = assemble_matrix(a, spaces.κ_trial, spaces.κ_test)
    Mκ = lu(Mκ)

    return FEData(mesh, spaces, dofs, ν, ν₀, κᵥ, κᵥ₀, Mν, Mκ)
end

function update_ν!(fe_data::FEData, params::Parameters, b)
    spaces = fe_data.spaces

    # compute f^2 / ∂z(b)
    dΩ = fe_data.mesh.dΩ
    f = params.f
    l(v) = ∫( (f*(f/∂z(b)))*v )dΩ
    y = assemble_vector(l, spaces.ν_test)
    sol = fe_data.Mν\y

    # ν = maximum(1, f^2 / ∂z(b))
    was_modified = false  # bool to track if ν was modified
    T = eltype(sol)
    for i in eachindex(sol)
        ν_i_prev = fe_data.ν.free_values[i]  # for `was_modified`
        fe_data.ν.free_values[i] = max(one(T), sol[i])
        ν_i_prev != fe_data.ν.free_values[i] && (was_modified = true)
    end

    return fe_data, was_modified
end

function update_κᵥ!(fe_data::FEData, params::Parameters, b)
    spaces = fe_data.spaces

    # rhs nonzero where ∂z(b) < 0 (i.e. unstable stratification)
    stability(x) = x < 0 ? 1.0 : 0.0
    dΩ = fe_data.mesh.dΩ
    l(v) = ∫( (stability∘∂z(b))*v )dΩ
    y = assemble_vector(l, spaces.κ_test)
    sol = clamp.(fe_data.Mκ\y, 0.0, 1.0)  # have to clamp between 0 and 1 to avoid weird negative values

    # increase κᵥ where unstable
    unstable_count = 0    # debug
    threshold = 1         # minimum value of sol to consider unstable
    was_modified = false  # bool to track if κᵥ was modified
    for i in eachindex(sol)
        κᵥ_i_prev = fe_data.κᵥ.free_values[i]  # for `was_modified`
        if sol[i] ≥ threshold  # unstable
            # set κᵥ to κᶜ
            fe_data.κᵥ.free_values[i] = params.κᶜ
            unstable_count += 1
        else  # stable
            # reset
            fe_data.κᵥ.free_values[i] = fe_data.κᵥ₀.free_values[i]
        end
        κᵥ_i_prev != fe_data.κᵥ.free_values[i] && (was_modified = true)
    end
    @info "κᵥ updated: $(unstable_count) unstable nodes"

    # # compute ∂z(b) in κ space
    # dΩ = fe_data.mesh.dΩ
    # l(v) = ∫( ∂z(b)*v )dΩ
    # y = assemble_vector(l, spaces.κ_test)
    # bz = clamp.(fe_data.Aκ\y, -1, Inf)  # force -1 ≤ ∂z(b) ≤ ∞
    # # bz = fe_data.Aκ\y

    # # increase κᵥ where unstable
    # unstable_count = 0  # debug
    # was_modified = false  # bool to track if κᵥ was modified
    # for i in eachindex(bz)
    #     κᵥ_i_prev = fe_data.κᵥ.free_values[i]  # for `was_modified`
    #     if bz[i] ≤ 0  # unstable
    #         # increase κᵥ linearly with -∂z(b) up to κᶜ
    #         fe_data.κᵥ.free_values[i] = fe_data.κᵥ₀.free_values[i]*(1 + bz[i]) - params.κᶜ*bz[i]
    #         # # set κᵥ to κᶜ
    #         # fe_data.κᵥ.free_values[i] = params.κᶜ
    #         unstable_count += 1
    #     else  # stable
    #         # reset
    #         fe_data.κᵥ.free_values[i] = fe_data.κᵥ₀.free_values[i]
    #     end
    #     κᵥ_i_prev != fe_data.κᵥ.free_values[i] && (was_modified = true)
    # end
    # @info "Updating κᵥ: $(unstable_count) unstable nodes"

    return fe_data, was_modified
end