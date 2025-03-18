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

function Model(arch::AbstractArchitecture, params::Parameters, mesh::Mesh, inversion::InversionToolkit)
    # empty evolution toolkit for now
    evolution = nothing
    state = rest_state(mesh)
    return Model(arch, params, mesh, inversion, evolution, state)
end

# function solve!(model::Model, t_final)
#     i_step = model.state.t ÷ model.params.Δt + 1
#     n_steps = t_final ÷ model.params.Δt
#     for i ∈ i_step:n_steps
#         take_step!(model)

#         if mod(i, n_steps ÷ 100) == 0
#             @info @sprintf("average ∂z(b) = %1.5e", sum(∫(model.params.N² + ∂z(model.state.b))model.mesh.dΩ)/sum(∫(1)model.mesh.dΩ))
#         end
#     end
#     return model
# end

# function take_step!(model::Model)
#     evolve_advection!(model)
#     evolve_diffusion!(model)
#     invert!(model)
#     model.state.t += model.params.Δt
#     return model
# end

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
    model.state.b.free_values .= b.free_values
    return model
end

function invert!(model::Model)
    # run the iterative solver
    invert!(model.inversion, model.state.b)

    # sync solution to state
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