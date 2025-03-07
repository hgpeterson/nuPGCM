using JLD2
import JLD2.save

mutable struct State{V, P, T}
    u::V
    v::V
    w::V
    p::P
    b::V
    t::T
end

function State(u::V, v::V, w::V, p::V, b::V, t) where V <: CellField
    # assuming here that p.free_values is a Gridap.Arrays.LazyArray
    return State(u.free_values, v.free_values, w.free_values, p.free_values.args[1], b.free_values, t)
end

function rest_state(U, V, W, P, B, t)
    u = interpolate_everywhere(0, U)
    v = interpolate_everywhere(0, V)
    w = interpolate_everywhere(0, W)
    p = interpolate_everywhere(0, P) 
    b = interpolate_everywhere(0, B)
    return State(u, v, w, p, b, t)
end

function set_state!(state::State, mesh::Mesh, inversion_toolkit::InversionToolkit)
    x = on_architecture(CPU(), inversion_toolkit.solver.x[mesh.dofs.inv_p_inversion])
    nu = mesh.dofs.nu
    nv = mesh.dofs.nv
    nw = mesh.dofs.nw
    state.u .= x[1:nu]
    state.v .= x[nu+1:nu+nv]
    state.w .= x[nu+nv+1:nu+nv+nw]
    state.p .= x[nu+nv+nw+1:end]
    return state
end

function save(state::State; ofile="state.jld2")
    jldsave(ofile; state)
    @info "State saved to '$ofile'"
end

function load_state(ifile)
    d = jldopen(ifile, "r")
    state = d["state"]
    close(d)
    @info "State loaded from '$ifile'"
    return state
end