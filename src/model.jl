import Base: show

mutable struct State{U, P, B}
    u::U     # flow in x direction
    v::U     # flow in y direction
    w::U     # flow in z direction
    p::P     # pressure
    b::B     # buoyancy
    t::Real  # time
end

struct Model{A<:AbstractArchitecture, P<:Parameters, F<:Forcings, D<:FEData, 
             I<:InversionToolkit, E<:EvolutionToolkit, S<:State}
    arch::A
    params::P
    forcings::F
    fe_data::D
    inversion::I
    evolution::E
    state::S
end

function inversion_model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, 
                         fe_data::FEData, inversion::InversionToolkit)
    evolution = nothing # this model is only used for calculating the inversion, no need for evolution toolkit
    state = rest_state(fe_data.spaces)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state)
end

function rest_state_model(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, 
                          fe_data::FEData, inversion::InversionToolkit, evolution::EvolutionToolkit)
    state = rest_state(fe_data.spaces)
    return Model(arch, params, forcings, fe_data, inversion, evolution, state)
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

function run!(model::Model; n_steps, i_step=1, n_save=Inf, n_plot=Inf, advection=true)
    # unpack
    Δt = model.params.Δt
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b

    # save initial condition for comparison
    b0 = interpolate(0, model.fe_data.spaces.B_trial)
    b0.free_values .= b.free_values
    volume = sum(∫( 1 )*model.fe_data.mesh.dΩ) # volume of the domain

    # start timer
    t0 = time()

    # number of steps between info print
    n_info = min(10, max(div(n_steps, 100, RoundNearest), 1))
    @info "Beginning integration with" n_steps i_step n_save n_plot n_info

    # need to store a half-step buoyancy for advection
    b_half = interpolate(0, model.fe_data.spaces.B_trial)
    for i ∈ i_step:n_steps
        # Strang split evolution equation
        evolve_hdiffusion!(model)             # Δt/2 horizontal diffusion
        evolve_vdiffusion!(model)             # Δt/2 vertical diffusion
        if advection
            evolve_advection!(model, b_half)  # Δt advection
        end
        evolve_vdiffusion!(model)             # Δt/2 vertical diffusion
        evolve_hdiffusion!(model)             # Δt/2 horizontal diffusion
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
            msg *= @sprintf("time per step: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)/(i-i_step+1))...)
            msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)*(n_steps-i)/(i-i_step+1))...)
            msg *= @sprintf("|u|ₘₐₓ = %.1e, %.1e ≤ b ≤ %.1e\n", max(u_max, v_max, w_max), minimum([b.free_values; 0]), maximum([b.free_values; 0]))
            # msg *= @sprintf("V⁻¹ ∫ (b - b0) dx = %.16f\n", sum(∫(b - b0)*model.fe_data.mesh.dΩ)/volume)
            msg
            end
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

        if model.forcings.eddy_param
            K = 1
            α = model.params.α
            f = model.params.f
            νₘₐₓ = model.params.νₘₐₓ
            N² = model.params.N²

            b_background = interpolate_everywhere(x -> N²*x[3], model.fe_data.spaces.B_trial)
            bz = ∂z(b_background + b)

            # want to have ν ∼ K f² / (α*bz) but this is funky near α*bz = 0
            # instead of 1/(α*bz), use c2 / √(c1 + α²bz²) which is: 
            #   • 1 at α*bz = 1, 
            #   • νₘₐₓ at α*bz = 0, and 
            #   • goes like 1/(α*bz) as α*bz → ∞
            c1 = 1 / (νₘₐₓ^2 - 1)
            c2 = νₘₐₓ * √c1
            ν = K * (f * (f * (c2 / (sqrt∘(c1 + α^2 * bz * bz)))))

            A_inversion = build_A_inversion(model.fe_data, model.params, ν)
            perm = model.fe_data.dofs.p_inversion
            A_inversion = A_inversion[perm, perm]
            model.inversion.solver.A = on_architecture(model.arch, A_inversion)
            # keeping same preconditioner (1/h^dim)
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
    B_test = model.fe_data.spaces.B_test
    dΩ = model.fe_data.mesh.dΩ
    N² = model.params.N²
    Δt = model.params.Δt
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b
    solver_adv = model.evolution.solver_adv
    arch = architecture(solver_adv.y)
    b_diri = model.fe_data.spaces.b_diri

    # sync up flow with current buoyancy state
    invert!(model)

    # compute b_half
    l_half(d) = ∫( b*d - Δt/2*(u*∂x(b) + v*∂y(b) + w*(N² + ∂z(b)))*d )dΩ
    l_half_diri(d) = ∫( b_diri*d - Δt/2*(u*∂x(b_diri) + v*∂y(b_diri) + w*∂z(b_diri))*d )dΩ
    solver_adv.y .=  on_architecture(arch, assemble_vector(l_half, B_test)[p_b])
    solver_adv.y .-= on_architecture(arch, assemble_vector(l_half_diri, B_test)[p_b]) #TODO: for performance, would be nice to find a better way to handle dirichlet correction
    iterative_solve!(solver_adv)
    b_half.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

    # compute u_half, v_half, w_half, p_half
    invert!(model, b_half)

    # full step
    l_full(d) = ∫( b*d - Δt*(u*∂x(b_half) + v*∂y(b_half) + w*(N² + ∂z(b_half)))*d )dΩ
    l_full_diri(d) = ∫( b_diri*d - Δt*(u*∂x(b_diri) + v*∂y(b_diri) + w*∂z(b_diri))*d )dΩ
    solver_adv.y .= on_architecture(arch, assemble_vector(l_full, B_test)[p_b])
    solver_adv.y .-= on_architecture(arch, assemble_vector(l_full_diri, B_test)[p_b])
    iterative_solve!(solver_adv)

    # sync buoyancy to state
    b.free_values .= on_architecture(CPU(), solver_adv.x[inv_p_b])

    return model
end

function evolve_vdiffusion!(model::Model)
    if model.forcings.convection
        # model, was_modified = update_κᵥ!(model, model.state.b)

        # if was_modified
        #     @info "Vertical diffusivity κᵥ was modified, rebuilding vertical diffusion system"
            # A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(model.fe_data, model.params, model.fe_data.κᵥ)
            b_background = interpolate_everywhere(x -> model.params.N²*x[3], model.fe_data.spaces.B_trial)
            κᵥ = model.params.κᶜ*(1 + tanh∘(-10*(∂z(b_background + model.state.b))))/2 + model.forcings.κᵥ
            A_vdiff, B_vdiff, b_vdiff = build_vdiffusion_system(model.fe_data, model.params, κᵥ)
            perm = model.fe_data.dofs.p_b
            A_vdiff = A_vdiff[perm, perm]
            B_vdiff = B_vdiff[perm, :]
            b_vdiff = b_vdiff[perm]
            model.evolution.solver_vdiff.A = on_architecture(model.arch, A_vdiff)
            model.evolution.solver_vdiff.P = Diagonal(on_architecture(model.arch, Vector(1 ./ diag(A_vdiff))))
            model.evolution.B_vdiff = on_architecture(model.arch, B_vdiff)
            model.evolution.b_vdiff = on_architecture(model.arch, b_vdiff)
        # end
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

function update_κᵥ!(model::Model, b)
    _, was_modified = update_κᵥ!(model.fe_data, model.params, b)
    return model, was_modified
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
    nv = model.fe_data.dofs.nv
    nw = model.fe_data.dofs.nw
    model.state.u.free_values .= x[1:nu]
    model.state.v.free_values .= x[nu+1:nu+nv]
    model.state.w.free_values .= x[nu+nv+1:nu+nv+nw]
    model.state.p.free_values.args[1] .= x[nu+nv+nw+1:end]
    return model
end