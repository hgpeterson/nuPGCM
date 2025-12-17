mutable struct State{U, UB, P, B}
    u::U     # linear part of flow
    ub::UB   # bubble part of flow
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
    println(io, "├── ub: ", state.ub, " with ", length(state.ub.free_values), " DOFs")
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
    U = spaces.U
    UB = spaces.UB
    P = spaces.P
    B = spaces.B

    # define FE functions
    u = interpolate(VectorValue(0, 0, 0), U)
    ub = interpolate(VectorValue(0, 0, 0), UB)
    p = interpolate(0, P) 
    b = interpolate(0, B)

    return State(u, ub, p, b, t)
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
    ub = model.state.ub
    b = model.state.b

    # save initial condition for comparison
    b0 = interpolate(0, model.fe_data.spaces.B)
    b0.free_values .= b.free_values
    # volume = sum(∫( 1 )*model.fe_data.mesh.dΩ) # volume of the domain

    if model.forcings.eddy_param.is_on
        # store inversion matrix without friction to speed up re-builds
        A_part = build_A_inversion(model.fe_data, model.params, model.forcings.ν; frictionless_only=true) 
    end

    # start timer
    t0 = time()

    # number of steps between info print
    n_info = min(10, max(div(n_steps, 100, RoundNearest), 1))
    @info "Beginning integration with" n_steps i_step n_save n_plot n_info

    # need to store a half-step buoyancy for advection
    b_half = interpolate(0, model.fe_data.spaces.B)
    t_last_info = time()  # another timer for ETR
    for i ∈ i_step:n_steps

        # Strang split evolution equation
        @time "hdiff" evolve_hdiffusion!(model)             # Δt/2 horizontal diffusion
        @time "vdiff" evolve_vdiffusion!(model)             # Δt/2 vertical diffusion
        if advection
            @time "adv" evolve_advection!(model, b_half)  # Δt advection
        end
        @time "vdiff" evolve_vdiffusion!(model)             # Δt/2 vertical diffusion
        @time "hdiff" evolve_hdiffusion!(model)             # Δt/2 horizontal diffusion
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
        ub_max = maximum(abs.(ub.free_values))
        b_max = maximum(abs.(b.free_values))
        if maximum([u_max, ub_max, b_max]) > 1e3
            throw(ErrorException("Blow-up detected, stopping simulation"))
        end

        if mod(i, n_info) == 0
            t1 = time()
            t_step = (t1 - t_last_info)/n_info
            @info begin
            msg  = @sprintf("t = %f (i = %d/%d, Δt = %f)\n", model.state.t, i, n_steps, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t1-t0)...)
            if i > n_info  # skip ETR the first time since it will contain compilation time
                msg *= @sprintf("timestep duration ~ %.1e s\n", t_step)
                msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs(t_step*(n_steps - i))...)
            end
            msg *= @sprintf("|u|ₘₐₓ = %.1e, |ub|ₘₐₓ = %.1e\n", u_max, ub_max)
            msg *= @sprintf("%.1e ≤ b ≤ %.1e\n", minimum([b.free_values; 0]), maximum([b.free_values; 0]))
            # msg *= @sprintf("V⁻¹ ∫ (b - b0) dx = %.16f\n", sum(∫(b - b0)*model.fe_data.mesh.dΩ)/volume)
            msg
            end
            t_last_info = t1
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

        flush(stdout)
        flush(stderr)
    end
    return model
end

function evolve_advection!(model::Model, b_half)
    # unpack
    p_b = model.fe_data.dofs.p_b
    inv_p_b = model.fe_data.dofs.inv_p_b
    D = model.fe_data.spaces.D
    dΩ = model.fe_data.mesh.dΩ
    N² = model.params.N²
    Δt = model.params.Δt
    b = model.state.b
    solver_adv = model.evolution.solver_adv
    arch = architecture(solver_adv.y)
    b_diri = model.fe_data.spaces.b_diri

    # sync up flow with current buoyancy state
    @time "  invert" invert!(model)

    # compute b_half
    u = model.state.u + model.state.ub
    w = u⋅z⃗
    l_half(d) = ∫( b*d - Δt/2*(u⋅∇(b) + w*N²)*d )dΩ
    l_half_diri(d) = ∫( b_diri*d - Δt/2*(u⋅∇(b_diri))*d )dΩ
    @time "  build adv_rhs1" solver_adv.y .=  on_architecture(arch, assemble_vector(l_half, D)[p_b])
    @time "  build adv_rhs2" solver_adv.y .-= on_architecture(arch, assemble_vector(l_half_diri, D)[p_b]) #TODO: for performance, would be nice to find a better way to handle dirichlet correction
    iterative_solve!(solver_adv)
    b_half.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

    # compute u_half, v_half, w_half, p_half
    @time "  invert" invert!(model, b_half)

    # full step
    u = model.state.u + model.state.ub
    w = u⋅z⃗
    l_full(d) = ∫( b*d - Δt*(u⋅∇(b_half) + w*N²)*d )dΩ
    l_full_diri(d) = ∫( b_diri*d - Δt*(u⋅∇(b_diri))*d )dΩ
    @time "  build adv_rhs1" solver_adv.y .= on_architecture(arch, assemble_vector(l_full, D)[p_b])
    @time "  build adv_rhs2" solver_adv.y .-= on_architecture(arch, assemble_vector(l_full_diri, D)[p_b])
    iterative_solve!(solver_adv)

    # sync buoyancy to state
    b.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

    return model
end

function evolve_vdiffusion!(model::Model)
    if model.forcings.conv_param.is_on
        α = model.params.α
        N² = model.params.N²
        b = model.state.b
        αbz = α*N² + α*∂z(b)
        κᵥ = κᵥ_convection(model.forcings, αbz)
        @time "  build vdiff" A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(model.fe_data, model.params, model.forcings, κᵥ)
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

    # sync solution to state
    b.free_values .= on_architecture(CPU(),
                                solver_vdiff.x[model.fe_data.dofs.inv_p_b]
                                )

    return model
end

function evolve_hdiffusion!(model::Model)
    # solve
    evolve_hdiffusion!(model.evolution, model.state.b)

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
    nub = model.fe_data.dofs.nub
    model.state.u.free_values .= x[1:nu]
    model.state.ub.free_values .= x[nu+1:nu+nub]
    model.state.p.free_values.args[1] .= x[nu+nub+1:end]
    return model
end