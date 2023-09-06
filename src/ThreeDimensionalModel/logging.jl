"""
    save_setup(m, save_file)

Save .h5 file for parameters.
"""
function save_setup(m::ModelSetup3D, save_file)
    file = h5open(save_file, "w")
    write(file, "ε²", m.ε²)
    write(file, "μ", m.μ)
    write(file, "ϱ", m.ϱ)
    write(file, "Δt", m.Δt)
    write(file, "f", m.f)
    write(file, "β", m.β)
    write(file, "H", m.H.values)
    write(file, "τx", m.τx.values)
    write(file, "τy", m.τy.values)
    write(file, "ν", m.ν.values)
    write(file, "κ", m.κ.values)
    write(file, "p_sfc1", m.g_sfc1.p)
    write(file, "t_sfc1", m.g_sfc1.t)
    write(file, "e_sfc1", m.g_sfc1.e["bdy"])
    write(file, "p_sfc2", m.g_sfc2.p)
    write(file, "t_sfc2", m.g_sfc2.t)
    write(file, "e_sfc2", m.g_sfc2.e["bdy"])
    write(file, "p1", m.g1.p)
    write(file, "t1", m.g1.t)
    write(file, "e1_sfc", m.g1.e["sfc"])
    write(file, "e1_bot", m.g1.e["bot"])
    write(file, "p2", m.g2.p)
    write(file, "t2", m.g2.t)
    write(file, "e2_sfc", m.g2.e["sfc"])
    write(file, "e2_bot", m.g2.e["bot"])
    write(file, "p_col", m.g_col.p)
    write(file, "t_col", m.g_col.t)
    write(file, "e_col_sfc", m.g_col.e["sfc"])
    write(file, "e_col_bot", m.g_col.e["bot"])
    write(file, "advection", m.advection)
    close(file)
    println(save_file)

    # log 
    ofile = open("$out_folder/out.txt", "w")
    println("\n$out_folder/out.txt contents:\n")
    log_params(ofile, "3D νPGCM with Parameters:")
    log_params(ofile, @sprintf("ε  = %1.1e (δ = %1.1e)", sqrt(m.ε²), sqrt(2*m.ε²)))
    log_params(ofile, @sprintf("μ  = %1.1e", m.μ))
    log_params(ofile, @sprintf("ϱ  = %1.1e", m.ϱ))
    log_params(ofile, @sprintf("f₀ = %1.1e", m.f))
    log_params(ofile, @sprintf("β  = %1.1e", m.β))
    log_params(ofile, @sprintf("Δt = %1.1e", m.Δt))
    log_params(ofile, "\nResolution:")
    log_params(ofile, @sprintf("np = %d", m.g_sfc1.np))
    log_params(ofile, @sprintf("nσ = %d (σ[2] - σ[1] = %1.1e)", m.nσ, m.σ[2] - m.σ[1]))
    log_params(ofile, "\nKeywords:")
    if m.σ[2] - m.σ[1] == m.σ[3] - m.σ[2]
        log_params(ofile, "chebyshev = false")
    else
        log_params(ofile, "chebyshev = true")
    end
    log_params(ofile, "advection = $(m.advection)")
    close(ofile)
end
function save_setup(m::ModelSetup3D)
    save_setup(m, "$out_folder/setup.h5")
end

"""
    m = load_setup_3D(filename)

Load .h5 setup file given by `filename`.
"""
function load_setup_3D(filename)
    file = h5open(filename, "r")
    ε² = read(file, "ε²")
    μ = read(file, "μ")
    ϱ = read(file, "ϱ")
    Δt = read(file, "Δt")
    f = read(file, "f")
    β = read(file, "β")

    # grids
    p_sfc1 = read(file, "p_sfc1")
    t_sfc1 = read(file, "t_sfc1")
    e_sfc1 = read(file, "e_sfc1")
    g_sfc1 = Grid(Triangle(order=1), p_sfc1, t_sfc1, e_sfc1)

    p_sfc2 = read(file, "p_sfc2")
    t_sfc2 = read(file, "t_sfc2")
    e_sfc2 = read(file, "e_sfc2")
    g_sfc2 = Grid(Triangle(order=2), p_sfc2, t_sfc2, e_sfc2)

    p1 = read(file, "p1")
    t1 = read(file, "t1")
    e1_sfc = read(file, "e1_sfc")
    e1_bot = read(file, "e1_bot")
    g1 = Grid(Wedge(order=1), p1, t1, Dict("sfc"=>e1_sfc, "bot"=>e1_bot))

    p2 = read(file, "p2")
    t2 = read(file, "t2")
    e2_sfc = read(file, "e2_sfc")
    e2_bot = read(file, "e2_bot")
    g2 = Grid(Wedge(order=2), p2, t2, Dict("sfc"=>e2_sfc, "bot"=>e2_bot))

    p_col = read(file, "p_col")
    t_col = read(file, "t_col")
    e_col_sfc = read(file, "e_col_sfc")
    e_col_bot = read(file, "e_col_bot")
    g_col = Grid(Line(order=1), p_col, t_col, Dict("sfc"=>e_col_sfc, "bot"=>e_col_bot))

    # fields
    Hvals = read(file, "H")
    τxvals = read(file, "τx")
    τyvals = read(file, "τy")
    νvals = read(file, "ν")
    κvals = read(file, "κ")
    H = FEField(Hvals, g_sfc2)
    τx = FEField(τxvals, g_sfc2)
    τy = FEField(τyvals, g_sfc2)
    ν = FEField(νvals, g2)
    κ = FEField(κvals, g2)

    advection = read(file, "advection")
    close(file)
    return ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, ν, κ, g_sfc1, g_sfc2, g1, g2, g_col, advection=advection)
end

"""
    save_state(s, save_file

Save .h5 state file.
"""
function save_state(s::ModelState3D, save_file)
    file = h5open(save_file, "w")
    write(file, "b", s.b.values)
    write(file, "p2", s.b.g.p)
    write(file, "t2", s.b.g.t)
    write(file, "e2_sfc", s.b.g.e["sfc"])
    write(file, "e2_bot", s.b.g.e["bot"])
    write(file, "ωx", s.ωx.values)
    write(file, "ωy", s.ωy.values)
    write(file, "χx", s.χx.values)
    write(file, "χy", s.χy.values)
    write(file, "p1", s.ωx.g.p)
    write(file, "t1", s.ωx.g.t)
    write(file, "e1_sfc", s.ωx.g.e["sfc"])
    write(file, "e1_bot", s.ωx.g.e["bot"])
    write(file, "Ψ", s.Ψ.values)
    write(file, "p_sfc", s.Ψ.g.p)
    write(file, "t_sfc", s.Ψ.g.t)
    write(file, "e_sfc", s.Ψ.g.e["bdy"])
    write(file, "i", s.i)
    close(file)
    println(save_file)
end

"""
    s = load_state_3D(filename)

Load .h5 state file given by `filename`.
"""
function load_state_3D(filename)
    file = h5open(filename, "r")

    # b
    bvals = read(file, "b")
    p2 = read(file, "p2")
    t2 = read(file, "t2")
    e2_sfc = read(file, "e2_sfc")
    e2_bot = read(file, "e2_bot")
    g2 = Grid(Wedge(order=2), p2, t2, Dict("sfc"=>e2_sfc, "bot"=>e2_bot))
    b = FEField(bvals, g2)

    # ω, χ
    ωxvals = read(file, "ωx")
    ωyvals = read(file, "ωy")
    χxvals = read(file, "χx")
    χyvals = read(file, "χy")
    p1 = read(file, "p1")
    t1 = read(file, "t1")
    e1_sfc = read(file, "e1_sfc")
    e1_bot = read(file, "e1_bot")
    g1 = Grid(Wedge(order=1), p1, t1, Dict("sfc"=>e1_sfc, "bot"=>e1_bot))
    ωx = DGField(ωxvals, g1)
    ωy = DGField(ωyvals, g1)
    χx = DGField(χxvals, g1)
    χy = DGField(χyvals, g1)

    # Ψ
    Ψvals = read(file, "Ψ")
    p_sfc = read(file, "p_sfc")
    t_sfc = read(file, "t_sfc")
    e_sfc = read(file, "e_sfc")
    g_sfc = Grid(Triangle(order=1), p_sfc, t_sfc, e_sfc)
    Ψ = FEField(Ψvals, g_sfc)

    i = read(file, "i")

    close(file)

    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, i)
    return s
end