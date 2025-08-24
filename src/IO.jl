function save_state(model::Model, ofile)
    s = model.state
    jldsave(ofile; u=s.u.free_values, v=s.v.free_values, w=s.w.free_values, 
                   p=s.p.free_values.args[1], b=s.b.free_values, t=s.t)
    @info "Model state saved to '$ofile'"
end

function set_state_from_file!(s::State, ifile)
    d = jldopen(ifile, "r")
    s.u.free_values .= d["u"]
    s.v.free_values .= d["v"]
    s.w.free_values .= d["w"]
    s.p.free_values.args[1] .= d["p"]
    s.b.free_values .= d["b"]
    s.t = d["t"]
    close(d)
    @info "State set from '$ifile'"
    return s
end

function save_vtk(m::Model; ofile="$out_dir/data/state.vtu")
    s = m.state
    b_background = interpolate_everywhere(x -> m.params.N²*x[3], m.fed.spaces.B_trial)
    writevtk(m.fed.mesh.Ω, ofile, cellfields=[
        "u" => s.u, 
        "v" => s.v, 
        "w" => s.w, 
        "p" => s.p, 
        "b" => b_background + s.b,
    ])

    @info "VTK state saved to '$ofile'"
end