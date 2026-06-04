function save_state(model::Model, ofile)
    s = model.state
    t = isnothing(model.timestepper) ? 0.0 : model.timestepper.t[]
    jldsave(ofile; u=s.u, p=s.p, b=s.b, t=t)
    @info "Model state saved to '$ofile'"
end

function set_state_from_file!(m::Model, ifile)
    d = jldopen(ifile, "r")
    m.state.u .= d["u"]
    m.state.p .= d["p"]
    m.state.b .= d["b"]
    if !isnothing(m.timestepper)
        m.timestepper.t[] = d["t"]
    end
    close(d)
    @info "Model state set from '$ifile'"
    return m
end

"""
    save_vtk(model; ofile="...")

Write the current model state to a VTK file using Ferrite's `VTKGridFile`.
The file name is `ofile.vtu`.
"""
function save_vtk(m::Model; ofile="$out_dir/data/state")
    fe_data = m.fe_data
    grid    = fe_data.mesh.grid
    state   = m.state

    # reassemble the combined (u, p) DOF vector in dh_up ordering
    x_up = zeros(ndofs(fe_data.dh_up))
    x_up[fe_data.u_dof_indices] .= state.u
    x_up[fe_data.p_dof_indices] .= state.p

    VTKGridFile(ofile, grid) do vtk
        write_solution(vtk, fe_data.dh_up, x_up)
        write_solution(vtk, fe_data.dh_b,  state.b)
    end
    @info "VTK state saved to '$ofile.vtu'"
end
