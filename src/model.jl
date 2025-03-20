import Krylov: solve!
import Gridap: solve!

mutable struct State{U, P, B}
    u::U     # flow in x direction
    v::U     # flow in y direction
    w::U     # flow in z direction
    p::P     # pressure
    b::B     # buoyancy
    t::Real  # time
end

struct Model{A, P, M, I, E, S}
    arch::A
    params::P
    mesh::M
    inversion::I
    evolution::E
    state::S
end

function rest_state_model(arch::AbstractArchitecture, params::Parameters, mesh::Mesh, inversion::InversionToolkit, evolution::EvolutionToolkit)
    state = rest_state(mesh)
    return Model(arch, params, mesh, inversion, evolution, state)
end

function rest_state(mesh::Mesh; t=0.)
    # unpack
    U = mesh.spaces.X_trial[1]
    V = mesh.spaces.X_trial[2]
    W = mesh.spaces.X_trial[3]
    P = mesh.spaces.X_trial[4]
    B = mesh.spaces.B_trial

    # define FE functions
    u = interpolate_everywhere(0, U)
    v = interpolate_everywhere(0, V)
    w = interpolate_everywhere(0, W)
    p = interpolate_everywhere(0, P) 
    b = interpolate_everywhere(0, B)

    return State(u, v, w, p, b, t)
end

function set_b!(model::Model, b::Function)
    # interpolate function onto FE space
    b_fe = interpolate_everywhere(b, model.state.b.fe_space)

    model.state.b.free_values .= b_fe.free_values

    return model
end
function set_b!(model::Model, b::AbstractArray)
    model.state.b.free_values .= b
    return model
end

function solve!(model::Model, t_final)
    # unpack
    t = model.state.t
    Δt = model.params.Δt
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b

    # start timer
    t0 = time()

    # number of steps to take
    n_steps = t_final ÷ Δt

    # starting step number (just 1 if t = 0)
    i_step = t ÷ Δt + 1

    # need to store a half-step buoyancy for advection
    b_half = interpolate_everywhere(0, model.mesh.spaces.B_trial)
    for i ∈ i_step:n_steps
        evolve_advection!(model, b_half)

        evolve_diffusion!(model)

        invert!(model)

        t += Δt

        if mod(i, n_steps ÷ 100) == 0
            # @info @sprintf("average ∂z(b) = %1.5e", sum(∫(model.params.N² + ∂z(model.state.b))model.mesh.dΩ)/sum(∫(1)model.mesh.dΩ))

            u_max = maximum(abs.(u.free_values))
            v_max = maximum(abs.(v.free_values))
            w_max = maximum(abs.(w.free_values))
            t1 = time()
            @info begin
            msg  = @sprintf("t = %f (i = %d/%d, Δt = %f)\n", t, i, n_steps, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t1-t0)...)
            msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)*(n_steps-i)/(i-i_step+1))...)
            msg *= @sprintf("|u|ₘₐₓ = %.1e, %.1e ≤ b′ ≤ %.1e\n", max(u_max, v_max, w_max), minimum([b.free_values; 0]), maximum([b.free_values; 0]))
            msg
            end
        end
    end
    return model
end

function evolve_advection!(model::Model, b_half)
    # unpack
    arch = model.arch
    p_b = model.mesh.dofs.p_b
    inv_p_b = model.mesh.dofs.inv_p_b
    B_test = model.mesh.spaces.B_test
    dΩ = model.mesh.dΩ
    N² = model.params.N²
    Δt = model.params.Δt
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b
    evolution = model.evolution

    # get u_half, v_half, w_half, b_half
    l_half(d) = ∫( b*d - Δt/2*(u*∂x(b) + v*∂y(b) + w*(N² + ∂z(b)))*d )dΩ
    evolution.rhs_vector .= on_architecture(arch, assemble_vector(l_half, B_test)[p_b])
    evolve_advection!(evolution)
    b_half.free_values .= on_architecture(CPU(), evolution.solver.x[inv_p_b])
    invert!(model, b_half) # u, v, w, p are now updated to half-step values

    # full step
    l_full(d) = ∫( b*d - Δt*(u*∂x(b_half) + v*∂y(b_half) + w*(N² + ∂z(b_half)))*d )dΩ
    evolution.rhs_vector .= on_architecture(arch, assemble_vector(l_full, B_test)[p_b])
    evolve_advection!(evolution)

    # sync state
    b.free_values .= on_architecture(CPU(), evolution.solver.x[inv_p_b])

    return model
end

function evolve_diffusion!(model::Model)
    evolve_diffusion!(model.evolution, model.state.b)

    # sync solution to state
    # TODO: maybe give `state` an `arch` field instead of insisting on `CPU`?
    model.state.b.free_values .= on_architecture(CPU(), model.evolution.solver.x[model.mesh.dofs.inv_p_b]) 
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
    # TODO: maybe give `state` an `arch` field instead of insisting on `CPU`?
    x = on_architecture(CPU(), model.inversion.solver.x[model.mesh.dofs.inv_p_inversion])
    nu = model.mesh.dofs.nu
    nv = model.mesh.dofs.nv
    nw = model.mesh.dofs.nw
    model.state.u.free_values .= x[1:nu]
    model.state.v.free_values .= x[nu+1:nu+nv]
    model.state.w.free_values .= x[nu+nv+1:nu+nv+nw]
    model.state.p.free_values.args[1] .= x[nu+nv+nw+1:end]
    return model
end
