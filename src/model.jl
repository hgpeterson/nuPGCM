import Base: show

mutable struct State{U, P, B}
    u::U     # flow in x direction
    v::U     # flow in y direction
    w::U     # flow in z direction
    p::P     # pressure
    b::B     # buoyancy
    t::Real  # time
end

struct Model{A, P, F, I, E, S}
    arch::A
    params::P
    fed::F
    inversion::I
    evolution::E
    state::S
end

function inversion_model(arch::AbstractArchitecture, params::Parameters, fed::FEData, inversion::InversionToolkit)
    evolution = nothing # this model is only used for calculating the inversion, no need for evolution toolkit
    state = rest_state(fed.spaces)
    return Model(arch, params, fed, inversion, evolution, state)
end

function rest_state_model(arch::AbstractArchitecture, params::Parameters, fed::FEData, inversion::InversionToolkit, evolution::EvolutionToolkit)
    state = rest_state(fed.spaces)
    return Model(arch, params, fed, inversion, evolution, state)
end

function rest_state(spaces::Spaces; t=0.)
    # unpack
    U, V, W, P = get_U_V_W_P(spaces)
    B = spaces.B_trial

    # define FE functions
    u = interpolate(0, U)
    v = interpolate(0, V)
    w = interpolate(0, W)
    p = interpolate(0, P) 
    b = interpolate(0, B)

    return State(u, v, w, p, b, t)
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

function run!(model::Model; n_steps, i_step=1, n_save=Inf, n_plot=Inf)
    # unpack
    Δt = model.params.Δt
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b

    # save initial condition for comparison
    b0 = interpolate(0, model.fed.spaces.B_trial)
    b0.free_values .= b.free_values
    volume = sum(∫( 1 )*model.fed.mesh.dΩ) # volume of the domain

    # start timer
    t0 = time()

    # number of steps between info print
    n_info = max(div(n_steps, 100, RoundNearest), 1)
    @info "Beginning integration with" n_steps i_step n_save n_plot n_info

    # need to store a half-step buoyancy for advection
    b_half = interpolate(0, model.fed.spaces.B_trial)
    for i ∈ i_step:n_steps
        # Strang split: Δt/2 diffusion, advection, Δt/2 diffusion
        evolve_diffusion!(model)
        evolve_advection!(model, b_half)
        evolve_diffusion!(model)
        model.state.t += Δt

        # blow-up -> stop
        u_max = maximum(abs.(u.free_values))
        v_max = maximum(abs.(v.free_values))
        w_max = maximum(abs.(w.free_values))
        b_max = maximum(abs.(b.free_values))
        if maximum([u_max, v_max, w_max, b_max]) > 1e3
            throw(ErrorException("Blow-up detected, stopping simulation"))
        end

        if mod(i, n_info) == 0
            t1 = time()
            @info begin
            msg  = @sprintf("t = %f (i = %d/%d, Δt = %f)\n", model.state.t, i, n_steps, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t1-t0)...)
            msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)*(n_steps-i)/(i-i_step+1))...)
            msg *= @sprintf("|u|ₘₐₓ = %.1e, %.1e ≤ b′ ≤ %.1e\n", max(u_max, v_max, w_max), minimum([b.free_values; 0]), maximum([b.free_values; 0]))
            msg *= @sprintf("V⁻¹ ∫ (b - b0) dx = %.16f\n", sum(∫(b - b0)*model.fed.mesh.dΩ)/volume)
            # msg *= @sprintf("V⁻¹ ∫ (∇⋅u⃗)^2 dx = %.16f\n", sum(∫( (∂x(u) + ∂y(v) + ∂z(w))*(∂x(u) + ∂y(v) + ∂z(w)) )*model.mesh.dΩ)/volume)
            msg
            end
            flush(stdout)
            flush(stderr)
        end

        if mod(i, n_save) == 0
            invert!(model) # sync flow with buoyancy state
            save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i))
            save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, i))
        end

        if mod(i, n_plot) == 0
            invert!(model) # sync flow with buoyancy state
            # sim_plots(model, model.state.t)
            plot_slice(model.state.u, model.state.b, model.params.N²; bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Zonal flow $u$",      fname=@sprintf("%s/images/u_channel_basin_xslice_%03d.png", out_dir, i))
            plot_slice(model.state.v, model.state.b, model.params.N²; bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_channel_basin_xslice_%03d.png", out_dir, i))
            plot_slice(model.state.w, model.state.b, model.params.N²; bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Vertical flow $w$",   fname=@sprintf("%s/images/w_channel_basin_xslice_%03d.png", out_dir, i))
        end
    end
    return model
end

function evolve_advection!(model::Model, b_half)
    # unpack
    p_b = model.fed.dofs.p_b
    inv_p_b = model.fed.dofs.inv_p_b
    B_test = model.fed.spaces.B_test
    dΩ = model.fed.mesh.dΩ
    N² = model.params.N²
    Δt = model.params.Δt
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b
    solver_adv = model.evolution.solver_adv
    b_diri = model.fed.spaces.b_diri

    # sync up flow with current buoyancy state
    invert!(model)

    # compute b_half
    l_half(d) = ∫( b*d - Δt/2*(u*∂x(b) + v*∂y(b) + w*(N² + ∂z(b)))*d )dΩ
    l_half_diri(d) = ∫( b_diri*d - Δt/2*(u*∂x(b_diri) + v*∂y(b_diri) + w*(N² + ∂z(b_diri)))*d )dΩ
    solver_adv.y .=  assemble_vector(l_half, B_test)[p_b]
    solver_adv.y .-= assemble_vector(l_half_diri, B_test)[p_b]
    iterative_solve!(solver_adv)
    b_half.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

    # compute u_half, v_half, w_half, p_half
    invert!(model, b_half)

    # full step
    l_full(d) = ∫( b*d - Δt*(u*∂x(b_half) + v*∂y(b_half) + w*(N² + ∂z(b_half)))*d )dΩ
    l_full_diri(d) = ∫( b_diri*d - Δt*(u*∂x(b_diri) + v*∂y(b_diri) + w*(N² + ∂z(b_diri)))*d )dΩ
    solver_adv.y .= assemble_vector(l_full, B_test)[p_b]
    solver_adv.y .-= assemble_vector(l_full_diri, B_test)[p_b]
    iterative_solve!(solver_adv)

    # sync buoyancy to state
    b.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

    return model
end

function evolve_diffusion!(model::Model)
    # solve
    evolve_diffusion!(model.evolution, model.state.b)

    # sync solution to state
    model.state.b.free_values .= model.evolution.solver_diff.x[model.fed.dofs.inv_p_b]
    return model
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
    # TODO: check that this works on GPU
    x = model.inversion.solver.x[model.fed.dofs.inv_p_inversion]
    nu = model.fed.dofs.nu
    nv = model.fed.dofs.nv
    nw = model.fed.dofs.nw
    model.state.u.free_values .= x[1:nu]
    model.state.v.free_values .= x[nu+1:nu+nv]
    model.state.w.free_values .= x[nu+nv+1:nu+nv+nw]
    model.state.p.free_values.args[1] .= x[nu+nv+nw+1:end]
    return model
end