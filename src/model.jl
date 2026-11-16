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
    apply!(b_sol, fe_data.ch_b)  # set the constrained DOFs
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

Copy the inversion solution on the free DOFs into `model.state.u` and
`model.state.p`, filling in the constrained DOFs.
"""
function sync_flow!(model::Model)
    fe_data = model.fe_data
    x_full  = model.inversion.x_full
    fill!(x_full, 0.0)
    x_full[fe_data.free_dofs] .= on_architecture(CPU(), model.inversion.solver.x)
    apply!(x_full, fe_data.ch_up)
    model.state.u .= x_full[fe_data.u_dof_indices]
    model.state.p .= x_full[fe_data.p_dof_indices]
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

    # the advection RHS is assembled on the CPU, from velocities in dh_up ordering
    b_prev_cpu = on_architecture(CPU(), b_prev)
    x_up      = to_up_vec(fe_data, on_architecture(CPU(), model.state.u))
    x_up_prev = to_up_vec(fe_data, on_architecture(CPU(), u_prev))

    @ctime "  build rhs_adv" rhs_adv = build_rhs_adv(fe_data, params,
                                                        x_up, b_cpu,
                                                        x_up_prev, b_prev_cpu,
                                                        timestepper)

    θ  = evolution_parameter(params, timestepper)
    Δt = effective_timestep(timestepper)
    y  = rhs_adv .+ θ .* on_architecture(CPU(), evolution.rhs_diff) .+
         Δt .* on_architecture(CPU(), evolution.rhs_flux) .+
         on_architecture(CPU(), evolution.f_bc)
    condense_rhs!(y, ch_b)
    evolution.solver.y .= on_architecture(arch, y)

    @ctime "  solve evol sys" iterative_solve!(evolution.solver)

    b_cpu = on_architecture(CPU(), evolution.solver.x)
    apply!(b_cpu, ch_b)      # set the constrained DOFs
    model.state.b .= b_cpu
    return model
end

"""
    run!(model; i_start=0, n_info=10, n_save=Inf, advection=true,
         n_precond=50, precond_growth=1.4)

Integrate `model` to `t_stop`, printing a status every `n_info` steps and saving
the state every `n_save` steps.

With the eddy parameterization on, the inversion preconditioner is rebuilt every
`n_precond` steps, and whenever a solve fails or takes `precond_growth` times the
iterations of the first solve after the last rebuild.
"""
function run!(model::Model; i_start=0, n_info=10, n_save=Inf, advection=true,
              n_precond=50, precond_growth=1.4)
    u = model.state.u
    b = model.state.b
    timestepper = model.timestepper

    @info "Beginning integration with" n_save n_info
    status(timestepper)

    h_min = minimum(compute_h_cells(model.fe_data.mesh))

    # for reporting the range of the total buoyancy b + N²z
    z_b    = b_dof_z(model.fe_data)
    b_full = similar(b)

    eddy_on   = model.forcings.eddy_param.is_on
    iterative = model.inversion.solver.P isa BlockTriangularPreconditioner
    if eddy_on
        update_eddy_A!(model)
        iterative && rebuild_preconditioner!(model)
        invert!(model)
    end

    if i_start == 0
        save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i_start))
        save_vtk(model,   ofile=@sprintf("%s/data/state_%016d", out_dir, i_start))
    end

    # iterations of the first inversion after the last preconditioner rebuild (0: not yet known)
    niter_ref = 0

    u_prev = copy(u)
    b_prev = copy(b)
    u_curr = copy(u)
    b_curr = copy(b)

    t₀ = t_last_info = time()
    i  = i_start + 1
    while timestepper.t[] < timestepper.t_stop
        @ctime "full step:" begin

        update_Δt!(timestepper, model.fe_data, u, h_min)
        Δt = timestepper.Δt[]

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

        if eddy_on && advection
            update_eddy_A!(model)
            if iterative
                stats = model.inversion.solver.workspace.stats
                grown = niter_ref > 0 && stats.niter > precond_growth * niter_ref
                if !stats.solved || grown || mod(i, n_precond) == 0
                    @ctime "  rebuild precond" rebuild_preconditioner!(model)
                    niter_ref = 0
                elseif niter_ref == 0
                    niter_ref = stats.niter
                end
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
            @. b_full = b + model.params.N² * z_b
            msg *= @sprintf("%.3e ≤ b + N²z ≤ %.3e\n", minimum(b_full), maximum(b_full))
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

function to_up_vec(fe_data::FEData, u_state::AbstractVector)
    x = zeros(ndofs(fe_data.dh_up))
    x[fe_data.u_dof_indices] .= u_state
    return x
end

"""
    rebuild_preconditioner!(model)

Rebuild the inversion preconditioner for the current eddy viscosity `ν(b)`.
"""
function rebuild_preconditioner!(model::Model)
    inversion = model.inversion
    inversion.solver.P = BlockTriangularPreconditioner(model.arch, model.fe_data, model.params,
                                                       model.forcings, inversion.lhs_cache.A_cond,
                                                       model.state.b)
    return model
end

"""
    update_eddy_A!(model)

Update the inversion matrix in place for the current eddy viscosity `ν(b)`.
"""
function update_eddy_A!(model::Model)
    lhs = model.inversion.lhs_cache
    @ctime "  build A_visc" build_A_visc!(lhs.A, lhs.A⁰, model.fe_data, model.params,
                                          model.forcings.eddy_param, model.state.b, lhs.nzidx_up)
    @ctime "  condense A" refresh_A_cond!(lhs)
    solver = model.inversion.solver
    @ctime "  update solver.A" update_A!(solver.A, lhs.A_cond, lhs.gpu_perm)
    solver.P isa Factorization && lu!(solver.P, lhs.A_cond)
    return model
end
