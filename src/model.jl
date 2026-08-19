struct State
    u::Vector{Float64}   # velocity DOFs (length nu)
    p::Vector{Float64}   # pressure DOFs (length np)
    b::Vector{Float64}   # buoyancy DOFs (length nb)
end

function Base.summary(state::State)
    t = typeof(state)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, state::State)
    println(io, summary(state), ":")
    println(io, "├── u: $(length(state.u)) DOFs")
    println(io, "├── p: $(length(state.p)) DOFs")
      print(io, "└── b: $(length(state.b)) DOFs")
end

struct Model{A<:AbstractArchitecture, P<:Parameters, F<:Forcings, D<:FEData,
             I<:InversionToolkit, E<:Union{EvolutionToolkit,Nothing},
             T<:Union{AbstractTimestepper,Nothing}}
    arch::A
    params::P
    forcings::F
    fe_data::D
    inversion::I
    evolution::E
    state::State
    timestepper::T
end

function Base.summary(model::Model)
    t = typeof(model)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, model::Model)
    println(io, summary(model), ":")
    println(io, "├── arch: ", model.arch)
    println(io, "├── params: ", summary(model.params))
    println(io, "├── forcings: ", summary(model.forcings))
    println(io, "├── fe_data: ", summary(model.fe_data))
    println(io, "├── inversion: ", summary(model.inversion))
    println(io, "├── evolution: ", summary(model.evolution))
    println(io, "├── state: ", summary(model.state))
      print(io, "└── timestepper: ", summary(model.timestepper))
end

# inversion-only model (no time integration)
function Model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings,
               fe_data::FEData, inversion::InversionToolkit)
    state = rest_state(fe_data)
    return Model(arch, params, forcings, fe_data, inversion, nothing, state, nothing)
end

# full model
function Model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings,
               fe_data::FEData, inversion::InversionToolkit, evolution::EvolutionToolkit,
               timestepper::AbstractTimestepper)
    state = rest_state(fe_data)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state, timestepper)
end

function rest_state(fe_data::FEData)
    nu, np, nb = get_n_dofs(fe_data)
    return State(zeros(nu), zeros(np), zeros(nb))
end

"""
    set_b!(model, b::Function)
    set_b!(model, b::AbstractVector)

Set the buoyancy initial condition via L2 projection of a function or direct
assignment of a DOF vector.
"""
function set_b!(model::Model, b_fn::Function)
    fe_data = model.fe_data
    dh_b    = fe_data.dh_b
    _, _, cv_b = make_cell_values(fe_data)
    n_b  = getnbasefunctions(cv_b)

    # assemble projection RHS:  rhs[i] = ∫ b_fn(x) φ_i dΩ
    rhs = zeros(fe_data.nb)
    fₑ  = zeros(n_b)
    for cc in CellIterator(dh_b)
        reinit!(cv_b, cc)
        coords = getcoordinates(cc)
        fill!(fₑ, 0.0)
        for q in 1:getnquadpoints(cv_b)
            x  = spatial_coordinate(cv_b, q, coords)
            dΩ = getdetJdV(cv_b, q)
            bq = b_fn(x)
            for i in 1:n_b
                fₑ[i] += bq * shape_value(cv_b, q, i) * dΩ
            end
        end
        rhs[celldofs(cc)] .+= fₑ
    end

    # solve M * b = rhs  (L2 projection)
    M = build_M(fe_data)
    apply!(M, rhs, fe_data.ch_b)
    b_sol = M \ rhs
    apply!(b_sol, fe_data.ch_b)  # recover constrained (mirror) DOFs: b[channel_west] = b[channel_east]
    model.state.b .= b_sol
    return model
end
function set_b!(model::Model, b::AbstractVector)
    model.state.b .= b
    return model
end

####
#### Time integration
####

function invert!(model::Model)
    return invert!(model, model.state.b)
end
function invert!(model::Model, b_vec::AbstractVector)
    invert!(model.inversion, b_vec)
    sync_flow!(model)
    return model
end

"""
    sync_flow!(model)

Copy the solver's *reduced* solution into `model.state.u` and `model.state.p`.

The solve is over free DOFs only (see [`condense_system`](@ref)), so the result
is first scattered into a full-length `N_up` buffer at `fe_data.free_dofs`; then
`apply!` reconstructs the constrained DOFs from their affine combinations plus
inhomogeneities (for periodic BCs: `u[mirror] = u[image]`), and the full vector
is split into the u and p fields.
"""
function sync_flow!(model::Model)
    x_red  = on_architecture(CPU(), model.inversion.solver.x)
    x_full = model.inversion.x_full
    fill!(x_full, 0.0)
    x_full[model.fe_data.free_dofs] .= x_red
    apply!(x_full, model.inversion.ch_up)  # recover constrained (mirror) DOFs
    model.state.u .= x_full[model.fe_data.u_dof_indices]
    model.state.p .= x_full[model.fe_data.p_dof_indices]
    return model
end

function evolve!(model::Model, u_prev::AbstractVector, b_prev::AbstractVector)
    fe_data    = model.fe_data
    params     = model.params
    forcings   = model.forcings
    timestepper = model.timestepper
    evolution  = model.evolution
    arch       = evolution.arch
    ch_b       = fe_data.ch_b

    b_cpu = on_architecture(CPU(), model.state.b)
    update_evolution_LHS!(evolution, fe_data, params, forcings, timestepper, b_cpu)

    # assemble advection RHS on CPU (field evaluation requires CPU)
    # build_rhs_adv needs the velocity DOF vector in the combined dh_up ordering
    b_prev_cpu = on_architecture(CPU(), b_prev)
    x_up      = _to_up_vec(fe_data, on_architecture(CPU(), model.state.u))
    x_up_prev = _to_up_vec(fe_data, on_architecture(CPU(), u_prev))

    @ctime "  build rhs_adv" rhs_adv = build_rhs_adv(fe_data, params,
                                                        x_up, b_cpu,
                                                        x_up_prev, b_prev_cpu,
                                                        timestepper)

    # combine RHS, apply BCs, and solve
    θ  = evolution_parameter(params, timestepper)
    Δt = timestepper.Δt[]
    y  = rhs_adv .+ θ .* on_architecture(CPU(), evolution.rhs_diff) .+
         Δt .* on_architecture(CPU(), evolution.rhs_flux) .+
         on_architecture(CPU(), evolution.f_bc)
    _condense_rhs!(y, ch_b)  # merge mirror rows into image rows, zero all constrained DOFs
    evolution.solver.y .= on_architecture(arch, y)

    @ctime "  solve evol sys" iterative_solve!(evolution.solver)

    b_cpu = on_architecture(CPU(), evolution.solver.x)
    apply!(b_cpu, ch_b)      # recover image DOFs and enforce surface Dirichlet BC
    model.state.b .= b_cpu
    return model
end

# specialise for BDF1 (no u_prev / b_prev needed)
function evolve!(model::Model, ::Nothing, ::Nothing)
    return evolve!(model, model.state.u, model.state.b)
end

function run!(model::Model; i_start=0, n_info=10, n_save=Inf, n_plot=Inf, advection=true,
              n_precond=50, precond_growth=1.4)
    u = model.state.u
    b = model.state.b
    timestepper = model.timestepper

    @info "Beginning integration with" n_save n_plot n_info
    status(timestepper)

    h_cells = compute_h_cells(model.fe_data.mesh)
    h_min   = minimum(h_cells)

    # initial eddy ν sync
    if model.forcings.eddy_param.is_on
        _update_eddy_A!(model)
        rebuild_preconditioner!(model)   # P was built at b = 0; resync it with the real ν(b)
        invert!(model)
    end

    if i_start == 0
        save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i_start))
        save_vtk(model,   ofile=@sprintf("%s/data/state_%016d", out_dir, i_start))
    end

    # iteration count of the first solve after the most recent preconditioner
    # rebuild; 0 means "not yet measured" (see the refresh block in the loop)
    niter_ref = 0

    u_prev = copy(u)
    b_prev = copy(b)
    u_curr = copy(u)
    b_curr = copy(b)

    t₀ = t_last_info = time()
    i  = i_start + 1
    while timestepper.t[] < timestepper.t_stop
        @ctime "full step:" begin

        update_Δt!(timestepper, model.fe_data, u, h_cells)
        Δt = timestepper.Δt[]

        if i == i_start + 2 && typeof(timestepper) <: BDF2
            collect_evolution_LHS!(model.evolution, model.params, model.forcings,
                                   timestepper, model.fe_data.ch_b; Kᵥ_changed=false)
        end

        u_curr .= u
        b_curr .= b

        evolve!(model, u_prev, b_prev)
        invert!(model)
        update_t!(timestepper)

        u_max = maximum(abs, u)
        b_max = maximum(abs, b)
        if max(u_max, b_max) > 1e3 || any(isnan, u) || any(isnan, b)
            throw(ErrorException("Blow-up detected, stopping simulation"))
        end

        u_prev .= u_curr
        b_prev .= b_curr

        if model.forcings.eddy_param.is_on && advection
            _update_eddy_A!(model)
            # Preconditioner refresh cadence.
            #
            # A fixed interval alone does not work. The staleness measurement behind
            # `n_precond` was made on a spun-up state, where b drifts slowly; during
            # the initial adjustment ν(b) moves fast enough to blow past `itmax`
            # within a few steps, and each such failure costs a full itmax solve
            # (~20 s) against a ~1.3 s rebuild.
            #
            # So drive it off the iteration count instead: remember what the solver
            # needed on the first solve after the last rebuild, and rebuild again
            # once it has grown by `precond_growth`, or if a solve outright failed.
            # The fixed interval stays as a backstop.
            stats  = model.inversion.solver.workspace.stats
            niter  = stats.niter
            solved = stats.solved
            grown  = niter_ref > 0 && niter > precond_growth * niter_ref
            if !solved || grown || (n_precond < Inf && mod(i, n_precond) == 0)
                @debug begin
                    why = !solved ? "solve failed" :
                          grown   ? @sprintf("iterations grew %d → %d", niter_ref, niter) :
                                    "periodic"
                    "rebuilding inversion preconditioner ($why)"
                end
                @ctime "  rebuild precond" rebuild_preconditioner!(model)
                niter_ref = 0        # next successful solve sets the new baseline
            elseif niter_ref == 0 && solved
                niter_ref = niter
            end
        end

        if mod(i, n_info) == 0
            t₁ = time()
            t_step = (t₁ - t_last_info) / n_info
            @info begin
            msg  = @sprintf("t = %.3e/%.3e (i = %d, Δt = %.3e)\n", timestepper.t[], timestepper.t_stop, i, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t₁ - t₀)...)
            if i > n_info
                msg *= @sprintf("timestep duration ~ %.3e s\n", t_step)
                msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n",
                                hrs_mins_secs(t_step * Int64((timestepper.t_stop - timestepper.t[]) ÷ Δt))...)
            end
            msg *= @sprintf("|u|ₘₐₓ = %.3e, CFL Δt ≈ %.3e\n", u_max, h_min / u_max)
            msg *= @sprintf("%.3e ≤ b ≤ %.3e, |db/dt|ₘₐₓ = %.3e\n",
                            minimum(b), maximum(b), maximum(abs, (b .- b_prev) ./ Δt))
            msg
            end
            t_last_info = t₁
        end

        if mod(i, n_save) == 0
            save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i))
            save_vtk(model,   ofile=@sprintf("%s/data/state_%016d", out_dir, i))
        end

        i += 1
        flush(stdout); flush(stderr)
        end
    end
    return model
end

function _to_up_vec(fe_data::FEData, u_state::AbstractVector)
    x = zeros(ndofs(fe_data.dh_up))
    x[fe_data.u_dof_indices] .= u_state
    return x
end

"""
    rebuild_preconditioner!(model)

Rebuild the inversion preconditioner from the current `ν(b)`.

`_update_eddy_A!` refreshes the *operator* every timestep but not the
preconditioner, which is correct for the constant `Diagonal(1/h³)` and wrong for
anything built from `ν`. Cahouet--Chabard's `Mν = ∫pq/(2α²ε²ν)` is exactly such a
thing (its other term, `Kf = ∫(1/|f|)∇p·∇q`, is ν-independent and needs no
refresh, but the whole preconditioner is rebuilt here for simplicity — setup is
~1.3 s at `h = 4e-2`).

No-op unless `InversionToolkit` was given a `NamedTuple` preconditioner spec and
the eddy parameterization is on (with fixed `ν` the operator never changes, so
neither does the preconditioner).

Call frequency is set by `run!`'s `n_precond`; see [`RefreshablePreconditioner`](@ref)
for the measured staleness that motivates the default of 50.
"""
function rebuild_preconditioner!(model::Model)
    inv_tk = model.inversion
    spec   = inv_tk.precond_spec
    spec === nothing && return model
    inv_tk.lhs_cache === nothing && return model    # ν fixed ⇒ nothing to refresh
    P = inv_tk.solver.P
    P isa RefreshablePreconditioner || return model
    P_new, _ = build_preconditioner(spec, model.arch, model.fe_data, model.params,
                                    model.forcings, inv_tk.lhs_cache.A_cond, nothing;
                                    b_vec = model.state.b, h = inv_tk.h_med)
    P.inner = P_new
    return model
end

"""
    _update_eddy_A!(model)

Refresh the inversion LHS for the current eddy viscosity `ν(b)`. Only the
viscous block changes; the static Coriolis/pressure/divergence blocks, the
condensation scatter map, and the diagonal preconditioner's mesh scale `h`
are all reused from `model.inversion.lhs_cache` (built once in
`InversionToolkit`), so this touches only `nzval` arrays — no sparse matrix
is (re)allocated. `f_bc` is left untouched: it is a fixed function of the
velocity Dirichlet inhomogeneities (all zero for this model, see
`InversionLHSCache`'s `g`), independent of `ν`.
"""
function _update_eddy_A!(model::Model)
    lhs = model.inversion.lhs_cache
    @ctime "  build A_visc" build_A_visc!(lhs.A, lhs.A⁰, model.fe_data, model.params,
                                          model.forcings.eddy_param, model.state.b, lhs.nzidx_up)
    @ctime "  condense A" refresh_A_cond!(lhs)   # writes lhs.A_cond in place
    @ctime "  update solver.A" update_A!(model.inversion.solver.A, lhs.A_cond, lhs.gpu_perm)
    return model
end
