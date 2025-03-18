function save_state(model::Model, ofile)
    s = model.state
    jldsave(ofile; u=s.u.free_values, v=s.v.free_values, w=s.w.free_values, 
                   p=s.p.free_values.args[1], b=s.b.free_values, t=s.t)
    @info "Model state saved to '$ofile'"
end

function set_state_from_file!(model::Model, ifile)
    d = jldopen(ifile, "r")
    model.state.u.free_values .= d["u"]
    model.state.v.free_values .= d["v"]
    model.state.w.free_values .= d["w"]
    model.state.p.free_values.args[1] .= d["p"]
    model.state.b.free_values .= d["b"]
    model.state.t = d["t"]
    close(d)
    @info "Model state set from '$ifile'"
    return model
end