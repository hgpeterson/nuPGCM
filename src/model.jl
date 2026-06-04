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
    model.state.b .= M \ rhs
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
    invert!(model.inversion, on_architecture(CPU(), b_vec))
    sync_flow!(model)
    return model
end

function sync_flow!(model::Model)
    x = on_architecture(CPU(), model.inversion.solver.x)
    model.state.u .= x[model.fe_data.u_dof_indices]
    model.state.p .= x[model.fe_data.p_dof_indices]
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

    if forcings.conv_param.is_on
        b_cpu = on_architecture(CPU(), model.state.b)
        @ctime "  build Kᵥ" begin
            Kᵥ_new = build_Kᵥ_conv(fe_data, params, forcings, b_cpu)
            evolution.Kᵥ.nzval .= Kᵥ_new.nzval
        end
        @ctime "  build rhs_diff" begin
            rhs_diff_new = build_rhs_diff_conv(params, fe_data, forcings, b_cpu)
            evolution.rhs_diff .= on_architecture(arch, rhs_diff_new)
        end
        collect_evolution_LHS!(evolution, params, forcings, timestepper, ch_b)
    end

    # assemble advection RHS on CPU (field evaluation requires CPU)
    # build_rhs_adv needs the velocity DOF vector in the combined dh_up ordering
    b_cpu      = on_architecture(CPU(), model.state.b)
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
    apply!(y, ch_b)
    evolution.solver.y .= on_architecture(arch, y)

    @ctime "  solve evol sys" iterative_solve!(evolution.solver)

    model.state.b .= on_architecture(CPU(), evolution.solver.x)
    return model
end

# specialise for BDF1 (no u_prev / b_prev needed)
function evolve!(model::Model, ::Nothing, ::Nothing)
    return evolve!(model, model.state.u, model.state.b)
end

function run!(model::Model; i_start=0, n_info=10, n_save=Inf, n_plot=Inf, advection=true)
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
        invert!(model)
    end

    if i_start == 0
        save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i_start))
        save_vtk(model,   ofile=@sprintf("%s/data/state_%016d", out_dir, i_start))
    end

    u_prev = copy(u)
    b_prev = copy(b)
    u_curr = copy(u)
    b_curr = copy(b)

    t₀ = t_last_info = time()
    i  = i_start + 1
    while timestepper.t[] < timestepper.t_stop
        @ctime "full step:" begin

        update_Δt!(timestepper, u, h_cells)
        Δt = timestepper.Δt[]

        if i == i_start + 2 && typeof(timestepper) <: BDF2
            collect_evolution_LHS!(model.evolution, model.params, model.forcings,
                                   timestepper, model.fe_data.ch_b)
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

        if model.forcings.eddy_param.is_on && advection && mod(i, 10) == 0
            _update_eddy_A!(model)
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

function _update_eddy_A!(model::Model)
    A_new = build_A_inversion(model.fe_data, model.params,
                               model.forcings.eddy_param, model.state.b)
    f_bc = zeros(size(A_new, 1))
    apply!(A_new, f_bc, model.fe_data.ch_up)

    # reuse the same diagonal preconditioner (h-scaled)
    p, t = get_p_t(model.fe_data.mesh)
    edges, _, _ = all_edges(t)
    hs = sort([norm(p[edges[i,1],:] - p[edges[i,2],:]) for i in axes(edges,1)])
    h  = hs[length(hs) ÷ 2]
    P  = Diagonal(on_architecture(model.arch, fill(1/h^3, size(A_new,1))))

    model.inversion.solver.A = on_architecture(model.arch, A_new)
    model.inversion.solver.P = P
    return model
end
