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

    if model.forcings.eddy_param.is_on
        # store inversion matrix without friction to speed up re-builds
        A_part = build_A_inversion(model.fe_data, model.params, model.forcings.ν; frictionless_only=true) 
    end

    # number of steps between info print
    n_info = min(10, max(div(n_steps, 100, RoundNearest), 1))
    @info "Beginning integration with" n_steps i_step n_save n_plot n_info

    # need to store a half-step buoyancy for advection (only used if model.evolution.order == 2)
    b_half = interpolate(0, model.fe_data.spaces.B_trial)

    # save buoyancy from previous timestep to check convergence
    b_prev = copy(model.state.b.free_values)

    ### DEBUG
    θ = Δt * model.params.α^2 * model.params.ε^2 / model.params.μϱ
    κ = model.forcings.κᵥ
    B_trial = model.fe_data.spaces.B_trial
    B_test = model.fe_data.spaces.B_test
    b_diri = model.fe_data.spaces.b_diri
    dΩ = model.fe_data.mesh.dΩ
    a(b, d) = ∫( b*d + θ*(κ*(∇(b)⋅∇(d))) )dΩ  # warning: ignoring N²
    A = assemble_matrix(a, B_trial, B_test)
    @time "lu(A_evolution)" A = lu(A)
    rhs_diri = assemble_vector(d->a(b_diri, d), B_test)

    # start timers
    t₀ = t_last_info = time()
    for i ∈ i_step:n_steps

        @ctime "full step:" begin

        # # Strang split evolution equation
        # @ctime "hdiff" evolve_hdiffusion!(model)             # Δt/2 horizontal diffusion
        # @ctime "vdiff" evolve_vdiffusion!(model)             # Δt/2 vertical diffusion
        # if advection
        #     @ctime "adv" evolve_advection!(model, b_half)  # Δt advection
        # end
        # @ctime "hdiff" evolve_vdiffusion!(model)             # Δt/2 vertical diffusion
        # @ctime "vdiff" evolve_hdiffusion!(model)             # Δt/2 horizontal diffusion

        ### DEBUG
        invert!(model)
        u = model.state.u
        w = u⋅z⃗
        b = model.state.b
        N² = model.params.N²
        # rhs  = assemble_vector(d->∫( b*d )dΩ, B_test)
        rhs  = assemble_vector(d->∫( b*d - Δt*(u⋅∇(b))*d )dΩ, B_test)
        # rhs  = assemble_vector(d->∫( b*d - Δt*(u⋅∇(b) + w*N²)*d )dΩ, B_test)
        # rhs  = assemble_vector(d->∫( b*d + Δt*b*(u⋅∇(d)) - Δt*(w*N²)*d )dΩ, B_test)
        rhs -= rhs_diri
        model.state.b.free_values .= A\rhs

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
            msg *= @sprintf("|u|ₘₐₓ = %.3e, %.3e ≤ b ≤ %.3e\n", u_max, minimum(b.free_values), maximum(b.free_values))
            msg *= @sprintf("|db/dt|ₘₐₓ = %.3e\n", maximum(abs.(b.free_values - b_prev)/Δt))
            msg
            end
            t_last_info = t₁
        end

        if mod(i, n_save) == 0
            invert!(model) # sync flow with buoyancy state
            save_state(model, @sprintf("%s/data/state_%016d.jld2", out_dir, i))
            save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, i))
        end

        if mod(i, n_plot) == 0
            invert!(model) # sync flow with buoyancy state
            sim_plots(model, model.state.t)
        end

        # set b_prev to current b
        b_prev .= b.free_values

        flush(stdout)
        flush(stderr)
        end
    end
    return model
end

function evolve_advection!(model::Model, b_half)
    # unpack
    p_b = model.fe_data.dofs.p_b
    inv_p_b = model.fe_data.dofs.inv_p_b
    B_test = model.fe_data.spaces.B_test
    dΩ = model.fe_data.mesh.dΩ
    N² = model.params.N²
    Δt = model.params.Δt
    u = model.state.u
    b = model.state.b
    solver_adv = model.evolution.solver_adv
    arch = architecture(solver_adv.y)
    b_diri = model.fe_data.spaces.b_diri
    order = model.evolution.order

    w = u⋅z⃗

    if order == 1 # Forward Euler
        # sync up flow with current buoyancy state
        @ctime "  invert" invert!(model)

        # compute b
        δb = b - b_diri
        # l(d) = ∫( δb*d - Δt*(u⋅∇(δb) + w*N²)*d )dΩ
        l(d) = ∫( δb*d - Δt*(u⋅∇(b) + w*N²)*d )dΩ
        # l(d) = ∫( -Δt*(u⋅∇(b) + w*N²)*d )dΩ
        @ctime "  build adv_rhs" solver_adv.y .=  on_architecture(arch, assemble_vector(l, B_test)[p_b])
        @ctime "  adv" iterative_solve!(solver_adv)

        # ### DEBUG
        # # b_new = model.evolution.solver_adv.x[model.fe_data.dofs.inv_p_b]
        # b_new = model.state.b.free_values + model.evolution.solver_adv.x[model.fe_data.dofs.inv_p_b]
        # b_new = FEFunction(model.fe_data.spaces.B_trial, b_new)
        # dbdt = (b_new - model.state.b)/model.params.Δt
        # writevtk(model.fe_data.mesh.Ω, "$out_dir/data/adv.vtu", cellfields=["dbdt" => dbdt])
        # @info "$out_dir/data/adv.vtu"

        # sync buoyancy to state
        b.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])
        # b.free_values .+= on_architecture(CPU(), solver_adv.x[inv_p_b])
    elseif order == 2 # RK2
        # sync up flow with current buoyancy state
        @ctime "  invert1" invert!(model)

        # compute b_half
        δb = b - b_diri
        l_half(d) = ∫( δb*d - Δt/2*(u⋅∇(δb) + w*N²)*d )dΩ
        @ctime "  build adv_rhs1" solver_adv.y .=  on_architecture(arch, assemble_vector(l_half, B_test)[p_b])
        @ctime "  adv1" iterative_solve!(solver_adv)
        b_half.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

        # compute u_half, v_half, w_half, p_half
        @ctime "  invert2" invert!(model, b_half)

        # full step
        δb_half = b_half - b_diri
        l_full(d) = ∫( δb*d - Δt*(u⋅∇(δb_half) + w*N²)*d )dΩ
        @ctime "  build adv_rhs2" solver_adv.y .= on_architecture(arch, assemble_vector(l_full, B_test)[p_b])
        @ctime "  adv2" iterative_solve!(solver_adv)

        # sync buoyancy to state
        b.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])
    else
        throw(ArgumentError("order must be 1 or 2"))
    end

    return model
end

function evolve_vdiffusion!(model::Model)
    if model.forcings.conv_param.is_on
        α = model.params.α
        N² = model.params.N²
        b = model.state.b
        order = model.evolution.order
        αbz = α*N² + α*∂z(b)
        κᵥ = κᵥ_convection(model.forcings, αbz)
        A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(model.fe_data, model.params, model.forcings, κᵥ; order)
        perm = model.fe_data.dofs.p_b
        A_vdiff = A_vdiff[perm, perm]
        B_vdiff = B_vdiff[perm, :]
        b_vdiff = b_vdiff[perm]
        model.evolution.solver_vdiff.A = on_architecture(model.arch, A_vdiff)
        model.evolution.solver_vdiff.P = Diagonal(on_architecture(model.arch, Vector(1 ./ diag(A_vdiff))))
        model.evolution.B_vdiff = on_architecture(model.arch, B_vdiff)
        model.evolution.b_vdiff = on_architecture(model.arch, b_vdiff)
    end

    # calculate rhs vector
    arch = model.arch
    solver_vdiff = model.evolution.solver_vdiff
    B_vdiff = model.evolution.B_vdiff
    b_vdiff = model.evolution.b_vdiff
    b = model.state.b
    solver_vdiff.y .= B_vdiff*on_architecture(arch, b.free_values) + b_vdiff

    # solve
    iterative_solve!(solver_vdiff)

    # ### DEBUG
    # b_new = model.evolution.solver_vdiff.x[model.fe_data.dofs.inv_p_b]
    # b_new = FEFunction(model.fe_data.spaces.B_trial, b_new)
    # dbdt = (b_new - model.state.b)/model.params.Δt
    # writevtk(model.fe_data.mesh.Ω, "$out_dir/data/vdiff.vtu", cellfields=["dbdt" => dbdt])
    # @info "$out_dir/data/vdiff.vtu"

    # sync solution to state
    b.free_values .= on_architecture(CPU(),
                                solver_vdiff.x[model.fe_data.dofs.inv_p_b]
                                )

    return model
end

function evolve_hdiffusion!(model::Model)
    # solve
    evolve_hdiffusion!(model.evolution, model.state.b)

    # ### DEBUG
    # b_new = model.evolution.solver_hdiff.x[model.fe_data.dofs.inv_p_b]
    # b_new = FEFunction(model.fe_data.spaces.B_trial, b_new)
    # dbdt = (b_new - model.state.b)/model.params.Δt
    # writevtk(model.fe_data.mesh.Ω, "$out_dir/data/hdiff.vtu", cellfields=["dbdt" => dbdt])
    # @info "$out_dir/data/hdiff.vtu"

    # sync solution to state
    model.state.b.free_values .= on_architecture(CPU(),
                                    model.evolution.solver_hdiff.x[model.fe_data.dofs.inv_p_b]
                                 )
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
    x = on_architecture(CPU(), model.inversion.solver.x[model.fe_data.dofs.inv_p_inversion])
    nu = model.fe_data.dofs.nu
    model.state.u.free_values .= x[1:nu]
    model.state.p.free_values.args[1] .= x[nu+1:end]
    return model
end