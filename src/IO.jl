function save_state(model::Model, ofile)
    s = model.state
    if isnothing(model.timestepper) 
        t = 0
    else
        t = model.timestepper.t[]
    end
    jldsave(ofile; u=s.u.free_values, p=s.p.free_values.args[1], b=s.b.free_values, t=t)
    @info "Model state saved to '$ofile'"
end

function set_state_from_file!(m::Model, ifile)
    d = jldopen(ifile, "r")
    m.state.u.free_values .= d["u"]
    m.state.p.free_values.args[1] .= d["p"]
    m.state.b.free_values .= d["b"]
    if !isnothing(m.timestepper)
        m.timestepper.t[] = d["t"]
    end
    close(d)
    @info "Model state set from '$ifile'"
    return m
end

function save_vtk(m::Model; ofile="$out_dir/data/state.vtu")
    s = m.state
    α = m.params.α
    N² = m.params.N²
    b = m.state.b
    B_trial = m.fe_data.spaces.B_trial
    b_full = interpolate_everywhere(x->N²*x[3], B_trial) + b
    αbz = α*N² + α*∂z(b)
    if m.forcings.eddy_param.is_on
        ν = ν_eddy(m.forcings.eddy_param, αbz)
    else
        ν = m.forcings.ν
    end
    if m.forcings.conv_param.is_on
        κᵥ = κᵥ_convection(m.forcings, αbz)
    else
        κᵥ = m.forcings.κᵥ
    end
    if isnothing(m.timestepper)
        t = 0
    else
        t = m.timestepper.t[]
    end
    # IMPORTANT: must have order = 2 for quadratic velocities!
    writevtk(m.fe_data.mesh.Ω, ofile, order=2, cellfields=[
        "u" => s.u, 
        "p" => s.p, 
        "b" => b_full,
        "alpha*b_z" => αbz,
        "nu" => ν,
        "kappa_v" => κᵥ,
        "t" => t,
    ])
    @info "VTK state saved to '$ofile'"
end
