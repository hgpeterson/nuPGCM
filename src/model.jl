struct State{U, P, B}
    u::U  # flow
    p::P  # pressure
    b::B  # buoyancy
end

function Base.summary(state::State)
    t = typeof(state)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, state::State)
    println(io, summary(state), ":")
    println(io, "├── u: ", state.u, " with ", length(state.u.free_values), " DOFs")
    println(io, "├── p: ", state.p, " with ", length(state.p.free_values), " DOFs")
      print(io, "└── b: ", state.b, " with ", length(state.b.free_values), " DOFs")
end

struct Model{A<:AbstractArchitecture, P<:Parameters, F<:Forcings, D<:FEData, 
             I<:InversionToolkit, E<:Union{EvolutionToolkit,Nothing}, S<:State, T<:AbstractTimestepper}
    arch::A
    params::P
    forcings::F
    fe_data::D
    inversion::I
    evolution::E
    state::S
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

# inversion model
function Model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, 
               fe_data::FEData, inversion::InversionToolkit)
    # this model is only used for calculating the inversion, no need for evolution/timestep stuff
    evolution = nothing 
    timestepper = nothing
    state = rest_state(fe_data.spaces)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state, timestepper)
end

# full model starting from rest
function Model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, 
               fe_data::FEData, inversion::InversionToolkit, evolution::EvolutionToolkit, 
               timestepper::AbstractTimestepper)
    state = rest_state(fe_data.spaces)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state, timestepper)
end

function rest_state(spaces::Spaces)
    # unpack
    U, P = spaces.X_trial
    B = spaces.B_trial

    # define FE functions
    u = interpolate(VectorValue(0, 0, 0), U)
    p = interpolate(0, P) 
    b = interpolate(0, B)

    return State(u, p, b)
end

function set_b!(model::Model, b::Function)
    # interpolate function onto FE space
    b_fe = interpolate(b, model.state.b.fe_space)

    model.state.b.free_values .= b_fe.free_values

    return model
end
function set_b!(model::Model, b::AbstractArray)
    model.state.b.free_values .= b
    return model
end

function run!(model::Model; n_info=10, n_save=Inf, n_plot=Inf, advection=true)
    # unpack
    u = model.state.u
    b = model.state.b
    timestepper = model.timestepper

    @info "Beginning integration with" n_save n_plot n_info 
    status(timestepper)

    # get resolution for CFL
    h_cells = compute_h_cells(model.fe_data.mesh)
    h_min = minimum(h_cells)

    if model.forcings.eddy_param.is_on
        # cache data needed to re-assemble inversion matrix
        X_trial = model.fe_data.spaces.X_trial
        X_test = model.fe_data.spaces.X_test
        dup = get_trial_fe_basis(X_trial)
        dvq = get_fe_basis(X_test)
        assembler = Gridap.SparseMatrixAssembler(X_trial, X_test)
        perm = model.fe_data.dofs.p_inversion
        iperm = model.fe_data.dofs.inv_p_inversion
        A_inversion = on_architecture(CPU(), model.inversion.solver.A)
        A_inversion = A_inversion[iperm, iperm]

        # # store inversion matrix without friction to speed up re-builds
        # A_part = build_A_inversion(model.fe_data, model.params, model.forcings.ν; frictionless_only=true) 
    end

    # store copies of previous and current u, b
    u_prev = FEFunction(model.fe_data.spaces.X_trial[1], copy(u.free_values))
    u_curr = FEFunction(model.fe_data.spaces.X_trial[1], copy(u.free_values))
    b_prev = FEFunction(model.fe_data.spaces.B_trial, copy(b.free_values))
    b_curr = FEFunction(model.fe_data.spaces.B_trial, copy(b.free_values))

    # start timers
    t₀ = t_last_info = time()
    i = 1
    while timestepper.t[] < timestepper.t_stop
        @ctime "full step:" begin

        update_Δt!(timestepper, u, model.fe_data.mesh.dΩ, h_cells)
        Δt = timestepper.Δt[]

        if i == 2 && typeof(timestepper) <: BDF2
            # need to do one step with BDF1, now we can switch to BDF2 if desired
            collect_evolution_LHS!(model.evolution, model.params, model.forcings, timestepper)
        end

        # sync current u, b before they update
        u_curr.free_values .= u.free_values
        b_curr.free_values .= b.free_values

        # do step
        evolve!(model, u_prev, b_prev)
        invert!(model)
        update_t!(timestepper)

        # blow-up -> stop
        u_max = maximum(abs.(u.free_values))
        b_max = maximum(abs.(b.free_values))
        if maximum([u_max, b_max]) > 1e3 || any(isnan.([u_max, b_max]))
            throw(ErrorException("Blow-up detected, stopping simulation"))
        end

        # set previous u, b to state before the update
        u_prev.free_values .= u_curr.free_values
        b_prev.free_values .= b_curr.free_values

        # update ν
        if model.forcings.eddy_param.is_on && advection && mod(i, 10) == 0
            α = model.params.α
            N² = model.params.N²
            b = model.state.b
            αbz = α*(N² + ∂z(b))
            ν = ν_eddy(model.forcings.eddy_param, αbz)
            # A_inversion = A_part + build_A_inversion(model.fe_data, model.params, ν; friction_only=true)
            build_A_inversion!(A_inversion, dup, dvq, assembler, model.fe_data, model.params, ν)
            model.inversion.solver.A = on_architecture(model.arch, A_inversion[perm, perm])
            # note: keeping same preconditioner (1/h^dim)
        end

        if mod(i, n_info) == 0
            t₁ = time()
            t_step = (t₁ - t_last_info)/n_info
            @info begin
            msg  = @sprintf("t = %.3e/%.3e (i = %d, Δt = %.3e)\n", timestepper.t[], timestepper.t_stop, i, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t₁-t₀)...)
            if i > n_info  # skip ETR the first time since it will contain compilation time
                msg *= @sprintf("timestep duration ~ %.3e s\n", t_step)
                msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", 
                                hrs_mins_secs(t_step*Int64((timestepper.t_stop - timestepper.t[]) ÷ Δt))...)
            end
            msg *= @sprintf("|u|ₘₐₓ = %.3e, CFL Δt ≈ %.3e\n", u_max, h_min/u_max)
            msg *= @sprintf("%.3e ≤ b_free ≤ %.3e, |db/dt|ₘₐₓ = %.3e\n", 
                            minimum(b.free_values), maximum(b.free_values), 
                            maximum(abs.(b.free_values - b_prev.free_values)/Δt))
            msg *= @sprintf("Memory usage: %.3e / %.3e GB\n", Sys.total_memory()/1e9 - Sys.free_memory()/1e9, Sys.total_memory()/1e9)
            msg *= @sprintf("Live heap: %.3e GB\n", Base.gc_live_bytes()/1e9)
            msg
            end
            t_last_info = t₁
        end

        if mod(i, n_save) == 0
            save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i))
            save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, i))
        end

        if mod(i, n_plot) == 0
            sim_plots(model, model.state.t)
        end

        # increment
        i += 1

        flush(stdout)
        flush(stderr)
        end
    end
    return model
end

function evolve!(model::Model, u_prev, b_prev)
    # unpack
    perm = model.fe_data.dofs.p_b
    inv_perm = model.fe_data.dofs.inv_p_b
    B_test = model.fe_data.spaces.B_test
    dΩ = model.fe_data.mesh.dΩ
    α = model.params.α
    N² = model.params.N²
    u = model.state.u
    b = model.state.b
    solver = model.evolution.solver
    arch = architecture(solver.y)

    # coefficient
    θ = evolution_parameter(model.params, model.timestepper)

    if model.forcings.conv_param.is_on
        # recompute κᵥ for convection
        αbz = α*(N² + ∂z(b))
        κᵥ = κᵥ_convection(model.forcings, αbz)

        # rebuild vertical diffusion components
        @ctime "  build Kᵥ" Kᵥ, rhsᵥ = build_Kᵥ(model.fe_data, κᵥ)  # TODO: cache
        # @ctime "  build Kᵥ" build_Kᵥ!(model, κᵥ)
        @ctime "  build rhs_diff" rhs_diff = build_rhs_diff(model.params, model.fe_data, κᵥ)
        Kᵥ = Kᵥ[perm, perm]
        rhsᵥ = rhsᵥ[perm]
        # model.evolution.Kᵥ .= model.evolution.Kᵥ_cache[perm, perm]
        # model.evolution.rhsᵥ .= model.evolution.rhsᵥ_cache[perm, perm]
        rhs_diff = rhs_diff[perm]
        rhsᵥ = on_architecture(arch, rhsᵥ)
        rhs_diff = on_architecture(arch, rhs_diff)
        model.evolution.rhsᵥ .= rhsᵥ
        model.evolution.rhs_diff .= rhs_diff
    else
        Kᵥ = model.evolution.Kᵥ
    end

    if model.forcings.conv_param.is_on || model.timestepper.adaptive
        # re-assemble matrix and preconditioner
        M = model.evolution.M
        Kₕ = model.evolution.Kₕ
        A = M + θ*(Kₕ + Kᵥ) 
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))

        # update model.evolution
        model.evolution.solver.A = on_architecture(arch, A)
        model.evolution.solver.P = P
    end

    # put together rhs and solve
    rhs_diff = model.evolution.rhs_diff
    rhs_flux = model.evolution.rhs_flux
    rhsₘ = model.evolution.rhsₘ
    rhsₕ = model.evolution.rhsₕ
    rhsᵥ = model.evolution.rhsᵥ
    w = u⋅z⃗
    w_prev = u_prev⋅z⃗
    @ctime "  build rhs_adv" rhs_adv = assemble_vector(
        d -> advection_lform(d, b, b_prev, u, u_prev, w, w_prev, N², dΩ, model.timestepper), 
        B_test)
    rhs_adv = rhs_adv[perm]
    rhs_adv  = on_architecture(arch, rhs_adv)
    # rhsᵥ  = on_architecture(arch, model.evolution.rhsᵥ)
    @. solver.y = rhs_adv + θ*rhs_diff + rhs_flux - (rhsₘ + θ*(rhsₕ + rhsᵥ))
    @ctime "  solve evol sys" iterative_solve!(solver)

    # sync buoyancy to state
    b.free_values .= on_architecture(CPU(), solver.x[inv_perm])

    return model
end

# function build_Kᵥ!(model::Model, κᵥ)
#     aᵥ(b, d) = ∫( κᵥ*∂z(b)*∂z(d) )dΩ
#     build_matrix_vector!(model.evolution.Kᵥ_cache, model.evolution.rhsᵥ_cache, aᵥ, model.evolution.assembler, model.fe_data.spaces.b_diri)
# end

function advection_lform(d, b, b_prev, u, u_prev, w, w_prev, N², dΩ, ts::BDF1)
    Δt = ts.Δt[]
    return ∫( ( b - Δt*( u⋅∇(b) + w*N² ) )*d )dΩ
end
#TODO: This assumes fixed Δt
function advection_lform(d, b, b_prev, u, u_prev, w, w_prev, N², dΩ, ts::BDF2)
    Δt = ts.Δt[]
    return ∫( ( 4/3*b - 1/3*b_prev - 2/3*Δt*( (2*u - u_prev)⋅∇(2*b - b_prev) + (2*w - w_prev)*N² ) )*d )dΩ
end

function invert!(model::Model)
    return invert!(model, model.state.b)
end
function invert!(model::Model, b)
    invert!(model.inversion, b)
    sync_flow!(model)
    return model
end

function sync_flow!(model::Model)
    x = on_architecture(CPU(), model.inversion.solver.x[model.fe_data.dofs.inv_p_inversion])
    nu = model.fe_data.dofs.nu
    model.state.u.free_values .= x[1:nu]
    model.state.p.free_values.args[1] .= x[nu+1:end]
    return model
end