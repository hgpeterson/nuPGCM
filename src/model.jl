mutable struct State{U, P, B}
    u::U     # flow
    p::P     # pressure
    b::B     # buoyancy
    t::Real  # time
end

function Base.summary(state::State)
    t = typeof(state)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, state::State)
    println(io, summary(state), ":")
    println(io, "├── u: ", state.u, " with ", length(state.u.free_values), " DOFs")
    println(io, "├── p: ", state.p, " with ", length(state.p.free_values), " DOFs")
    println(io, "├── b: ", state.b, " with ", length(state.b.free_values), " DOFs")
      print(io, "└── t: ", state.t)
end

struct Model{A<:AbstractArchitecture, P<:Parameters, F<:Forcings, D<:FEData, 
             I<:InversionToolkit, E<:Union{EvolutionToolkit,Nothing}, S<:State}
    arch::A
    params::P
    forcings::F
    fe_data::D
    inversion::I
    evolution::E
    state::S
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
      print(io, "└── state: ", summary(model.state))
end

# inversion model
function Model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, 
               fe_data::FEData, inversion::InversionToolkit)
    evolution = nothing # this model is only used for calculating the inversion, no need for evolution toolkit
    state = rest_state(fe_data.spaces)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state)
end

# full model starting from rest
function Model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, 
               fe_data::FEData, inversion::InversionToolkit, evolution::EvolutionToolkit)
    state = rest_state(fe_data.spaces)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state)
end

function rest_state(spaces::Spaces; t=0.)
    # unpack
    U, P = spaces.X_trial
    B = spaces.B_trial

    # define FE functions
    u = interpolate(VectorValue(0, 0, 0), U)
    p = interpolate(0, P) 
    b = interpolate(0, B)

    return State(u, p, b, t)
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

function run!(model::Model; n_steps, i_step=1, n_save=Inf, n_plot=Inf, advection=true)
    # unpack
    Δt = model.params.Δt
    u = model.state.u
    b = model.state.b

    # get resolution for CFL
    p, t = get_p_t(model.fe_data.mesh.model)
    edges, _, _ = all_edges(t)
    h_min = minimum([norm(p[edges[i, 1], :] - p[edges[i, 2], :]) for i ∈ axes(edges, 1)])

    if model.forcings.eddy_param.is_on
        # store inversion matrix without friction to speed up re-builds
        A_part = build_A_inversion(model.fe_data, model.params, model.forcings.ν; frictionless_only=true) 
    end

    # number of steps between info print
    n_info = min(10, max(div(n_steps, 100, RoundNearest), 1))
    @info "Beginning integration with" n_steps i_step n_save n_plot n_info

    # store previous timesteps
    u_prev = FEFunction(model.fe_data.spaces.X_trial[1], copy(model.state.u.free_values))
    b_prev = FEFunction(model.fe_data.spaces.B_trial, copy(model.state.b.free_values))

    # start timers
    t₀ = t_last_info = time()
    for i ∈ i_step:n_steps

        @ctime "full step:" begin

        evolve!(model, u_prev, b_prev)
        invert!(model)
        model.state.t += Δt

        if model.forcings.eddy_param.is_on && advection
            α = model.params.α
            N² = model.params.N²
            b = model.state.b
            αbz = α*N² + α*∂z(b)
            ν = ν_eddy(model.forcings.eddy_param, αbz)
            A_inversion = A_part + build_A_inversion(model.fe_data, model.params, ν; friction_only=true)
            perm = model.fe_data.dofs.p_inversion
            A_inversion = A_inversion[perm, perm]
            model.inversion.solver.A = on_architecture(model.arch, A_inversion)
            # note: keeping same preconditioner (1/h^dim)
        end

        # blow-up -> stop
        u_max = maximum(abs.(u.free_values))
        b_max = maximum(abs.(b.free_values))
        if maximum([u_max, b_max]) > 1e3 || any(isnan.([u_max, b_max]))
            throw(ErrorException("Blow-up detected, stopping simulation"))
        end

        if mod(i, n_info) == 0
            t₁ = time()
            t_step = (t₁ - t_last_info)/n_info
            @info begin
            msg  = @sprintf("t = %f (i = %d/%d, Δt = %.3e)\n", model.state.t, i, n_steps, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t₁-t₀)...)
            if i > n_info  # skip ETR the first time since it will contain compilation time
                msg *= @sprintf("timestep duration ~ %.3e s\n", t_step)
                msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs(t_step*(n_steps - i))...)
            end
            msg *= @sprintf("|u|ₘₐₓ = %.3e, CFL Δt ≈ %.3e\n", u_max, h_min/u_max)
            msg *= @sprintf("%.3e ≤ b ≤ %.3e, |db/dt|ₘₐₓ = %.3e\n", 
                            min(minimum(b.free_values), minimum(b.dirichlet_values)), 
                            max(maximum(b.free_values), maximum(b.dirichlet_values)), 
                            maximum(abs.(b.free_values - b_prev.free_values)/Δt))
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

        # set u_prev and b_prev to current u, b
        u_prev.free_values .= u.free_values
        b_prev.free_values .= b.free_values

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
    ε = model.params.ε
    α = model.params.α
    μϱ = model.params.μϱ
    N² = model.params.N²
    Δt = model.params.Δt
    u = model.state.u
    b = model.state.b
    solver = model.evolution.solver
    arch = architecture(solver.y)
    rhs_diff = model.evolution.rhs_diff
    rhsₘ = model.evolution.rhsₘ
    rhsₕ = model.evolution.rhsₕ
    rhsᵥ = model.evolution.rhsᵥ
    order = model.evolution.order

    # coefficient
    if order == 1
        # BDF1
        θ = Δt * α^2 * ε^2 / μϱ
    elseif order == 2
        # BDF2
        θ = 2/3 * Δt * α^2 * ε^2 / μϱ
    end

    if model.forcings.conv_param.is_on
        # recompute κᵥ for convection
        α = model.params.α
        N² = model.params.N²
        b = model.state.b
        αbz = α*N² + α*∂z(b)
        κᵥ = κᵥ_convection(model.forcings, αbz)

        # rebuild vertical diffusion components
        Kᵥ, rhsᵥ = build_Kᵥ(model.fe_data, κᵥ)
        rhs_diff = build_rhs_diff(model.params, model.forcings, model.fe_data, κᵥ)
        Kᵥ = Kᵥ[perm, perm]
        rhsᵥ = rhsᵥ[perm]
        rhs_diff = rhs_diff[perm]
        model.evolution.rhsᵥ .= rhsᵥ
        model.evolution.rhs_diff .= rhs_diff

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
    w = u⋅z⃗
    w_prev = u_prev⋅z⃗
    rhs_adv = assemble_vector(d -> advection_lform(d, b, b_prev, Δt, u, u_prev, w, w_prev, N², dΩ, Val(order)), B_test)
    rhs_adv = rhs_adv[perm]
    @ctime "  build evol_rhs" solver.y .= on_architecture(arch,
        rhs_adv + rhs_diff - (rhsₘ + θ*(rhsₕ + rhsᵥ))
    )
    @ctime "  solve evol sys" iterative_solve!(solver)

    # sync buoyancy to state
    b.free_values .= on_architecture(CPU(), solver.x[inv_perm])

    return model
end

function advection_lform(d, b, b_prev, Δt, u, u_prev, w, w_prev, N², dΩ, ::Val{1})
    # BDF1
    return ∫( ( b - Δt*( u⋅∇(b) + w*N² ) )*d )dΩ
end
function advection_lform(d, b, b_prev, Δt, u, u_prev, w, w_prev, N², dΩ, ::Val{2})
    # BDF2
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