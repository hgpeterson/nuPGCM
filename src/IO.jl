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
    b_background = interpolate_everywhere(x -> m.params.N²*x[3], m.fe_data.spaces.B_trial)
    K = 1
    α = m.params.α
    f = m.params.f
    νₘₐₓ = m.params.νₘₐₓ
    b = b_background + m.state.b
    filter(x) = x > 1 ? (x < νₘₐₓ ? x : νₘₐₓ) : one(x)
    ν = filter∘(K / α * (f * (f / ∂z(b))))
    κᵥ = m.params.κᶜ*(1 + tanh∘(-10*(∂z(b))))/2 + m.forcings.κᵥ
    # sx = -∂x(b)/∂z(b)
    # sy = -∂y(b)/∂z(b)
    # ub = -∂z(K*sx)  # error here because you can't take two derivatives?
    # vb = -∂z(K*sy)
    # wb = ∂x(K*sx) + ∂y(K*sy)
    writevtk(m.fe_data.mesh.Ω, ofile, cellfields=[
        "u" => s.u, 
        "v" => s.v, 
        "w" => s.w, 
        "p" => s.p, 
        "b" => b,
        "∂z(b)" => ∂z(b),
        "ν" => ν,
        "κᵥ" => κᵥ,
        # "sx" => sx,
        # "sy" => sy,
        # "ub" => ub,
        # "vb" => vb,
        # "wb" => wb,
    ])

    @info "VTK state saved to '$ofile'"
end