struct InversionToolkit{B, V, CUP, S<:IterativeSolverToolkit}
    B::B       # RHS coupling matrix (N_up × nb): maps buoyancy DOFs to (u,p) DOFs
    f_wind::V  # RHS from wind-stress surface integral
    f_bc::V    # RHS correction for inhomogeneous BCs (0 for homogeneous)
    ch_up::CUP # ConstraintHandler for setting constrained DOF values in invert!
    solver::S
end

function Base.summary(inv::InversionToolkit)
    t = typeof(inv)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, inv::InversionToolkit)
    println(io, summary(inv), ":")
    println(io, "├── B: ", summary(inv.B))
    println(io, "├── f_wind: ", summary(inv.f_wind))
    println(io, "├── f_bc: ", summary(inv.f_bc))
      print(io, "└── solver: ", summary(inv.solver))
end

"""
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; kwargs...)

Set up the toolkit for inverting the steady PG momentum equations

    -f(ẑ×u) + α²ε² ∇·(2ν ε(u)) - ∇p = (1/α) b ẑ
    ∇·u = 0
"""
function InversionToolkit(arch::AbstractArchitecture,
                          fe_data::FEData,
                          params::Parameters,
                          forcings::Forcings;
                          atol=1e-6, rtol=1e-6, itmax=0,
                          memory=20, history=true, verbose=false, restart=true)
    @info "Building inversion system..."

    A     = build_A_inversion(fe_data, params, forcings.ν)
    B     = build_B_inversion(fe_data, params)
    f_wind = build_f_wind(fe_data, params, forcings)

    # apply BCs: modifies A in-place, computes f_bc correction
    f_bc = zeros(size(A, 1))
    apply!(A, f_bc, fe_data.ch_up)

    # preconditioner
    if typeof(arch) == GPU || forcings.eddy_param.is_on
        p, t = get_p_t(fe_data.mesh)
        edges, _, _ = all_edges(t)
        hs = sort([norm(p[edges[i,1],:] - p[edges[i,2],:]) for i in axes(edges,1)])
        h  = hs[length(hs) ÷ 2]
        P  = Diagonal(on_architecture(arch, fill(1/h^3, size(A,1))))
    else
        @warn "LU-factoring inversion matrix with $(size(A,1)) DOFs..."
        @time "lu(A_inversion)" P = lu(A)
    end

    A      = on_architecture(arch, A)
    B      = on_architecture(arch, B)
    f_wind = on_architecture(arch, f_wind)
    f_bc   = on_architecture(arch, f_bc)
    print_memory_status(arch)

    N  = size(A, 1)
    T  = eltype(A)
    y  = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    workspace = Krylov.GmresWorkspace(N, N, VT; memory)
    workspace.x .= zero(T)

    verbose_int = verbose ? 1 : 0
    kwargs_dict = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax,
                       :history=>history, :verbose=>verbose_int, :restart=>restart)
    solver = IterativeSolverToolkit(A, P, y, workspace, kwargs_dict, "Inversion")

    return InversionToolkit(B, f_wind, f_bc, fe_data.ch_up, solver)
end

"""
    invert!(inv_tk, b_vec)

Solve the inversion system given buoyancy DOF vector `b_vec` (length nb).
The combined (u,p) solution is stored in `inv_tk.solver.x`.
"""
function invert!(inv_tk::InversionToolkit, b_vec::AbstractVector)
    arch = architecture(inv_tk.solver.A)
    y = on_architecture(CPU(), inv_tk.B) * b_vec .+
        on_architecture(CPU(), inv_tk.f_wind) .+
        on_architecture(CPU(), inv_tk.f_bc)
    apply!(y, inv_tk.ch_up)   # set constrained DOF values (0 for no-slip/periodic)
    inv_tk.solver.y .= on_architecture(arch, y)
    iterative_solve!(inv_tk.solver)
    return inv_tk
end

####
#### Matrix and vector builders
####

"""
    A = build_A_inversion(fe_data, params, ν)

Assemble the LHS matrix for the inversion problem:

    A[(u,v)] = ∫ 2α²ε² ν ε(u):ε(v) dΩ  (viscous)
             + ∫ f (ẑ×u)·v dΩ            (Coriolis)
             - ∫ (∇·v) p dΩ              (pressure gradient)
             + ∫ q (∇·u) dΩ              (divergence-free)
"""
function build_A_inversion(fe_data::FEData, params::Parameters, ν)
    dh_up = fe_data.dh_up
    cv_u, cv_p, _ = make_cell_values(fe_data)
    n_u   = getnbasefunctions(cv_u)
    n_p   = getnbasefunctions(cv_p)
    n_loc = n_u + n_p
    α²ε²  = params.α^2 * params.ε^2
    f_cor = params.f   # Coriolis parameter (function of x)

    A   = allocate_inversion_matrix(fe_data)
    asm = start_assemble(A)
    Ae  = zeros(n_loc, n_loc)

    for cc in CellIterator(dh_up)
        reinit!(cv_u, cc)
        reinit!(cv_p, cc)
        coords = getcoordinates(cc)
        fill!(Ae, 0.0)

        for q in 1:getnquadpoints(cv_u)
            x  = spatial_coordinate(cv_u, q, coords)
            ν_q = ν isa Function ? ν(x) : ν
            f_q = f_cor(x)
            dΩ  = getdetJdV(cv_u, q)

            for i in 1:n_u
                ε_i    = symmetric(shape_gradient(cv_u, q, i))
                φᵤ_i   = shape_value(cv_u, q, i)
                div_i  = tr(shape_gradient(cv_u, q, i))

                for j in 1:n_u
                    φⱼ = shape_value(cv_u, q, j)
                    # viscous: 2α²ε² ν ε(u):ε(v)
                    visc = 2α²ε² * ν_q * dcontract(ε_i, symmetric(shape_gradient(cv_u, q, j)))
                    # Coriolis: f (ẑ×u)·v,  ẑ×u = (-u₂, u₁, 0)
                    cori = f_q * (-φⱼ[2] * φᵤ_i[1] + φⱼ[1] * φᵤ_i[2])
                    Ae[i, j] += (visc + cori) * dΩ
                end

                for j in 1:n_p
                    # pressure gradient: -(∇·v) p
                    Ae[i, n_u + j] -= div_i * shape_value(cv_p, q, j) * dΩ
                end
            end

            for i in 1:n_p
                φ_p_i = shape_value(cv_p, q, i)
                for j in 1:n_u
                    # divergence-free: q (∇·u)
                    Ae[n_u + i, j] += φ_p_i * tr(shape_gradient(cv_u, q, j)) * dΩ
                end
            end
        end

        assemble!(asm, celldofs(cc), Ae)
    end
    return A
end

"""
    B = build_B_inversion(fe_data, params)

Assemble the buoyancy-to-velocity coupling matrix:

    B[u_i, b_j] = ∫ (1/α) φ_b_j (ẑ·φᵤ_i) dΩ

Returns a sparse matrix of size `(N_up × nb)` where N_up = ndofs(dh_up).
Only the u-rows are nonzero (pressure rows remain zero).
"""
function build_B_inversion(fe_data::FEData, params::Parameters)
    dh_up = fe_data.dh_up
    dh_b  = fe_data.dh_b
    cv_u, _, cv_b = make_cell_values(fe_data)
    n_u = getnbasefunctions(cv_u)
    n_b = getnbasefunctions(cv_b)
    α   = params.α
    N_up = ndofs(dh_up)
    nb   = fe_data.nb

    rows = Int[]; cols = Int[]; vals = Float64[]

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_u, cc_up)
        reinit!(cv_b, cc_b)
        dofs_up = celldofs(cc_up)
        dofs_b  = celldofs(cc_b)
        Be = zeros(n_u, n_b)

        for q in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q)
            for i in 1:n_u
                z_comp = shape_value(cv_u, q, i)[3]   # ẑ·φᵤ_i
                iszero(z_comp) && continue
                for j in 1:n_b
                    Be[i, j] += (1/α) * shape_value(cv_b, q, j) * z_comp * dΩ
                end
            end
        end

        u_range = dof_range(dh_up, :u)
        for (li, gi) in enumerate(dofs_up[u_range])
            for (lj, gj) in enumerate(dofs_b)
                v = Be[li, lj]
                iszero(v) && continue
                push!(rows, gi); push!(cols, gj); push!(vals, v)
            end
        end
    end

    return sparse(rows, cols, vals, N_up, nb)
end

"""
    A = build_A_inversion(fe_data, params, eddy_param, b_vec)

Assemble the inversion LHS with eddy viscosity computed at each quadrature point
from `∂z(b)` via `ν_eddy(eddy_param, α*(N² + ∂z(b)))`.
"""
function build_A_inversion(fe_data::FEData, params::Parameters,
                            eddy_param::EddyParameterization,
                            b_vec::AbstractVector)
    dh_up = fe_data.dh_up
    dh_b  = fe_data.dh_b
    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    n_u   = getnbasefunctions(cv_u)
    n_p   = getnbasefunctions(cv_p)
    n_loc = n_u + n_p
    α²ε²  = params.α^2 * params.ε^2
    f_cor = params.f
    α     = params.α
    N²    = params.N²

    A   = allocate_inversion_matrix(fe_data)
    asm = start_assemble(A)
    Ae  = zeros(n_loc, n_loc)

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_u, cc_up)
        reinit!(cv_p, cc_up)
        reinit!(cv_b, cc_b)
        coords  = getcoordinates(cc_up)
        local_b = b_vec[celldofs(cc_b)]
        fill!(Ae, 0.0)

        for q in 1:getnquadpoints(cv_u)
            x       = spatial_coordinate(cv_u, q, coords)
            ∂z_b_q  = function_gradient(cv_b, q, local_b)[3]
            αbz_q   = α * (N² + ∂z_b_q)
            ν_q     = ν_eddy(eddy_param, αbz_q)
            f_q     = f_cor(x)
            dΩ      = getdetJdV(cv_u, q)

            for i in 1:n_u
                ε_i   = symmetric(shape_gradient(cv_u, q, i))
                φᵤ_i  = shape_value(cv_u, q, i)
                div_i = tr(shape_gradient(cv_u, q, i))
                for j in 1:n_u
                    φⱼ = shape_value(cv_u, q, j)
                    visc = 2α²ε² * ν_q * dcontract(ε_i, symmetric(shape_gradient(cv_u, q, j)))
                    cori = f_q * (-φⱼ[2] * φᵤ_i[1] + φⱼ[1] * φᵤ_i[2])
                    Ae[i, j] += (visc + cori) * dΩ
                end
                for j in 1:n_p
                    Ae[i, n_u + j] -= div_i * shape_value(cv_p, q, j) * dΩ
                end
            end
            for i in 1:n_p
                φ_p_i = shape_value(cv_p, q, i)
                for j in 1:n_u
                    Ae[n_u + i, j] += φ_p_i * tr(shape_gradient(cv_u, q, j)) * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cc_up), Ae)
    end
    return A
end

"""
    f = build_f_wind(fe_data, params, forcings)

Assemble the wind-stress surface RHS:
`f[u_i] = ∫_Γ α (τˣ (x̂·φᵤ_i) + τʸ (ŷ·φᵤ_i)) dΓ`.
Returns a vector of length N_up; only u-rows on the surface facets are nonzero.
"""
function build_f_wind(fe_data::FEData, params::Parameters, forcings::Forcings)
    dh_up     = fe_data.dh_up
    fv_u, _   = make_facet_values(fe_data)
    n_u       = getnbasefunctions(fv_u)
    α         = params.α
    τˣ        = forcings.τˣ
    τʸ        = forcings.τʸ
    facetset  = fe_data.mesh.grid.facetsets[fe_data.mesh.surface_tag]
    u_range   = dof_range(dh_up, :u)

    f  = zeros(ndofs(dh_up))
    fₑ = zeros(n_u)

    for fc in FacetIterator(dh_up, facetset)
        reinit!(fv_u, fc)
        coords = getcoordinates(fc)
        fill!(fₑ, 0.0)
        for q in 1:getnquadpoints(fv_u)
            x  = spatial_coordinate(fv_u, q, coords)
            dΓ = getdetJdV(fv_u, q)
            for i in 1:n_u
                φᵤ_i = shape_value(fv_u, q, i)
                fₑ[i] += α * (τˣ(x) * φᵤ_i[1] + τʸ(x) * φᵤ_i[2]) * dΓ
            end
        end
        f[celldofs(fc)[u_range]] .+= fₑ
    end
    return f
end
