# write and read functions for sparse matrices 
import Base: write
function write(file::HDF5.File, A_str::AbstractString, A::SparseArrays.AbstractSparseMatrix)
    I, J, V = findnz(A)
    write(file, string(A_str, "_I"), I)
    write(file, string(A_str, "_J"), J)
    write(file, string(A_str, "_V"), V)
    write(file, string(A_str, "_m"), size(A, 1))
    write(file, string(A_str, "_n"), size(A, 2))
end
function read_sparse_matrix(file, A_str)
    I = read(file, string(A_str, "_I"))
    J = read(file, string(A_str, "_J"))
    V = read(file, string(A_str, "_V"))
    m = read(file, string(A_str, "_m"))
    n = read(file, string(A_str, "_n"))
    return sparse(I, J, V, m, n)
end

"""
    save_setup(m, save_file)

Save .h5 file for parameters.
"""
function save_setup(m::ModelSetup3D, save_file)
    file = h5open(save_file, "w")

    # Params
    write(file, "ε²", m.params.ε²)
    write(file, "μϱ", m.params.μϱ)
    write(file, "f", m.params.f)
    write(file, "β", m.params.β)

    # Geometry
    write(file, "H", m.geom.H.values)
    write(file, "Hx", m.geom.Hx.values)
    write(file, "Hy", m.geom.Hy.values)
    write(file, "p_sfc1", m.geom.g_sfc1.p)
    write(file, "t_sfc1", m.geom.g_sfc1.t)
    write(file, "e_sfc1", m.geom.g_sfc1.e["bdy"])
    write(file, "p_sfc2", m.geom.g_sfc2.p)
    write(file, "t_sfc2", m.geom.g_sfc2.t)
    write(file, "e_sfc2", m.geom.g_sfc2.e["bdy"])
    write(file, "in_nodes1", m.geom.in_nodes1)
    write(file, "in_nodes2", m.geom.in_nodes2)
    write(file, "p1", m.geom.g1.p)
    write(file, "t1", m.geom.g1.t)
    write(file, "e1_sfc", m.geom.g1.e["sfc"])
    write(file, "e1_bot", m.geom.g1.e["bot"])
    write(file, "p2", m.geom.g2.p)
    write(file, "t2", m.geom.g2.t)
    write(file, "e2_sfc", m.geom.g2.e["sfc"])
    write(file, "e2_bot", m.geom.g2.e["bot"])
    write(file, "e2_coast", m.geom.g2.e["coast"])
    write(file, "p_col", m.geom.g_col.p)
    write(file, "t_col", m.geom.g_col.t)
    write(file, "e_col_sfc", m.geom.g_col.e["sfc"])
    write(file, "e_col_bot", m.geom.g_col.e["bot"])
    write(file, "σ", m.geom.σ)
    write(file, "nσ", m.geom.nσ)

    # Forcing
    write(file, "τx", m.forcing.τx.values)
    write(file, "τy", m.forcing.τy.values)
    write(file, "τx_x", m.forcing.τx_x.values)
    write(file, "τx_y", m.forcing.τx_y.values)
    write(file, "τy_x", m.forcing.τy_x.values)
    write(file, "τy_y", m.forcing.τy_y.values)
    write(file, "ν", m.forcing.ν.values)
    write(file, "ν_bot", m.forcing.ν_bot.values)
    write(file, "κ", m.forcing.κ.values)

    # InversionComponents
    write(file, "Dx", m.inversion.Dx) # see above for sparse matrix
    write(file, "Dy", m.inversion.Dy)
    write(file, "M_bc", m.inversion.M_bc)
    write(file, "ωx_Ux", m.inversion.ωx_Ux)
    write(file, "ωy_Ux", m.inversion.ωy_Ux)
    write(file, "χx_Ux", m.inversion.χx_Ux)
    write(file, "χy_Ux", m.inversion.χy_Ux)
    write(file, "ωx_τx", m.inversion.ωx_τx)
    write(file, "ωy_τx", m.inversion.ωy_τx)
    write(file, "χx_τx", m.inversion.χx_τx)
    write(file, "χy_τx", m.inversion.χy_τx)

    # EvolutionComponents
    write(file, "HM", m.evolution.HM)
    write(file, "Ax1", Array(m.evolution.Ax1))
    write(file, "Ay1", Array(m.evolution.Ay1))
    write(file, "Ax2", Array(m.evolution.Ax2))
    write(file, "Ay2", Array(m.evolution.Ay2))
    write(file, "Ax_HM_SD", Array(m.evolution.Ax_HM_SD))
    write(file, "Ay_HM_SD", Array(m.evolution.Ay_HM_SD))
    write(file, "advection", m.evolution.advection)
    close(file)
    println(save_file)

    # log 
    ofile = open("$out_folder/out.txt", "w")
    println("\n$out_folder/out.txt contents:\n")
    log_params(ofile, "3D νPGCM with Parameters:")
    log_params(ofile, @sprintf("ε  = %1.1e (δ = %1.1e)", sqrt(m.params.ε²), sqrt(2*m.params.ε²)))
    log_params(ofile, @sprintf("μϱ = %1.1e", m.params.μϱ))
    log_params(ofile, @sprintf("f₀ = %1.1e", m.params.f))
    log_params(ofile, @sprintf("β  = %1.1e", m.params.β))
    log_params(ofile, "\nResolution:")
    log_params(ofile, @sprintf("np = %d", m.geom.g_sfc1.np))
    log_params(ofile, @sprintf("nσ = %d (σ[2] - σ[1] = %1.1e)", m.geom.nσ, m.geom.σ[2] - m.geom.σ[1]))
    log_params(ofile, "\nKeywords:")
    if m.geom.σ[2] - m.geom.σ[1] ≈ m.geom.σ[3] - m.geom.σ[2]
        log_params(ofile, "chebyshev = false")
    else
        log_params(ofile, "chebyshev = true")
    end
    log_params(ofile, "advection = $(m.evolution.advection)")
    close(ofile)
end
function save_setup(m::ModelSetup3D)
    save_setup(m, "$out_folder/data/setup.h5")
end

"""
    m = load_setup_3D(filename)

Load .h5 setup file given by `filename`.
"""
function load_setup_3D(filename)
    file = h5open(filename, "r")

    # Params
    println("Loading Params...")
    ε² = read(file, "ε²")
    μϱ = read(file, "μϱ")
    f = read(file, "f")
    β = read(file, "β")
    params = Params(; ε², μϱ, f, β)

    # Geometry
    println("Loading Geometry...")
    Hvals = read(file, "H")
    Hxvals = read(file, "Hx")
    Hyvals = read(file, "Hy")

    p_sfc1 = read(file, "p_sfc1")
    t_sfc1 = read(file, "t_sfc1")
    e_sfc1 = read(file, "e_sfc1")
    g_sfc1 = Grid(Triangle(order=1), p_sfc1, t_sfc1, e_sfc1)
    in_nodes1 = read(file, "in_nodes1")

    p_sfc2 = read(file, "p_sfc2")
    t_sfc2 = read(file, "t_sfc2")
    e_sfc2 = read(file, "e_sfc2")
    g_sfc2 = Grid(Triangle(order=2), p_sfc2, t_sfc2, e_sfc2)
    in_nodes2 = read(file, "in_nodes2")

    H = FEField(Hvals, g_sfc2)
    Hx = DGField(Hxvals, g_sfc1)
    Hy = DGField(Hyvals, g_sfc1)

    println("\tg1")
    p1 = read(file, "p1")
    t1 = read(file, "t1")
    e1_sfc = read(file, "e1_sfc")
    e1_bot = read(file, "e1_bot")
    g1 = Grid(Wedge(order=1), p1, t1, Dict("sfc"=>e1_sfc, "bot"=>e1_bot))

    println("\tg2")
    p2 = read(file, "p2")
    t2 = read(file, "t2")
    e2_sfc = read(file, "e2_sfc")
    e2_bot = read(file, "e2_bot")
    e2_coast = read(file, "e2_coast")
    g2 = Grid(Wedge(order=2), p2, t2, Dict("sfc"=>e2_sfc, "bot"=>e2_bot, "coast"=>e2_coast))

    p_col = read(file, "p_col")
    t_col = read(file, "t_col")
    e_col_sfc = read(file, "e_col_sfc")
    e_col_bot = read(file, "e_col_bot")
    g_col = Grid(Line(order=1), p_col, t_col, Dict("sfc"=>e_col_sfc, "bot"=>e_col_bot))
    σ = read(file, "σ")
    nσ = read(file, "nσ")

    g_sfc1_to_g1_map = build_g_sfc1_to_g1_map(g_sfc1, g1, nσ)
    coast_mask = build_coast_mask(g_sfc1, nσ)

    geom = Geometry(H, Hx, Hy, g_sfc1, g_sfc2, in_nodes1, in_nodes2, g1, g2, g_sfc1_to_g1_map, coast_mask, g_col, σ, nσ)

    # Forcing
    println("Loading Forcing...")
    τxvals = read(file, "τx")
    τyvals = read(file, "τy")
    τx_xvals = read(file, "τx_x")
    τx_yvals = read(file, "τx_y")
    τy_xvals = read(file, "τy_x")
    τy_yvals = read(file, "τy_y")
    νvals = read(file, "ν")
    ν_botvals = read(file, "ν_bot")
    κvals = read(file, "κ")
    τx = FEField(τxvals, g_sfc2)
    τy = FEField(τyvals, g_sfc2)
    τx_x = DGField(τx_xvals, g_sfc1)
    τx_y = DGField(τx_yvals, g_sfc1)
    τy_x = DGField(τy_xvals, g_sfc1)
    τy_y = DGField(τy_yvals, g_sfc1)
    ν = FEField(νvals, g2)
    ν_bot = FEField(ν_botvals, g_sfc1)
    κ = FEField(κvals, g2)
    forcing = Forcing(τx, τy, τx_x, τx_y, τy_x, τy_y, ν, ν_bot, κ)

    # InversionComponents
    println("Loading InversionComponents...")
    Dx = read_sparse_matrix(file, "Dx")
    Dy = read_sparse_matrix(file, "Dy")
    M_bc = read_sparse_matrix(file, "M_bc")
    ωx_Ux = read(file, "ωx_Ux")
    ωy_Ux = read(file, "ωy_Ux")
    χx_Ux = read(file, "χx_Ux")
    χy_Ux = read(file, "χy_Ux")
    ωx_τx = read(file, "ωx_τx")
    ωy_τx = read(file, "ωy_τx")
    χx_τx = read(file, "χx_τx")
    χy_τx = read(file, "χy_τx")
    baroclinic_LHSs = build_baroclinic_LHSs(params, geom, forcing)
    νωx_Ux_bot = ν_bot*FEField(ωx_Ux[:, 1], g_sfc1)
    νωy_Ux_bot = ν_bot*FEField(ωy_Ux[:, 1], g_sfc1)
    barotropic_LHS = build_barotropic_LHS(params, geom, νωx_Ux_bot, νωy_Ux_bot)
    νωx_τx_bot = ν_bot*FEField(ωx_τx[:, 1], g_sfc1)
    νωy_τx_bot = ν_bot*FEField(ωy_τx[:, 1], g_sfc1)
    τx1 = FEField(τx[1:g_sfc1.np], g_sfc1)
    τy1 = FEField(τy[1:g_sfc1.np], g_sfc1)
    νωx_τ_bot = τx1*νωx_τx_bot - τy1*νωy_τx_bot
    νωy_τ_bot = τx1*νωy_τx_bot + τy1*νωx_τx_bot
    barotropic_RHS_τ = build_barotropic_RHS_τ(params, geom, forcing, νωx_τ_bot, νωy_τ_bot)
    inversion = InversionComponents(Dx, Dy, baroclinic_LHSs, M_bc, ωx_Ux, ωy_Ux, χx_Ux, χy_Ux, barotropic_LHS, ωx_τx, ωy_τx, χx_τx, χy_τx, barotropic_RHS_τ)

    # EvolutionComponents
    println("Loading EvolutionComponents...")
    HM = read_sparse_matrix(file, "HM")
    K_cols = [build_K_col(σ, κ[get_col_inds(i, nσ)]) for i ∈ 1:g_sfc2.np]
    Ax1 = read(file, "Ax1")
    Ay1 = read(file, "Ay1")
    Ax2 = read(file, "Ax2")
    Ay2 = read(file, "Ay2")
    Ax_HM_SD = read(file, "Ax_HM_SD")
    Ay_HM_SD = read(file, "Ay_HM_SD")
    CUDA.memory_status()
    advection = read(file, "advection")
    evolution = EvolutionComponents(HM, K_cols, Ax1, Ay1, Ax2, Ay2, Ax_HM_SD, Ay_HM_SD, advection)

    close(file)
    return ModelSetup3D(params, geom, forcing, inversion, evolution)
end

"""
    save_state(s, save_file)

Save .h5 state file.
"""
function save_state(s::ModelState3D, save_file)
    file = h5open(save_file, "w")
    write(file, "b", s.b.values)
    write(file, "ωx", s.ωx.values)
    write(file, "ωy", s.ωy.values)
    write(file, "χx", s.χx.values)
    write(file, "χy", s.χy.values)
    write(file, "Ψ", s.Ψ.values)
    write(file, "t", s.t)
    close(file)
    println(save_file)
end

"""
    s = load_state_3D(m::ModelSetup3D, filename)

Load .h5 state file given by `filename`.
"""
function load_state_3D(m::ModelSetup3D, filename)
    file = h5open(filename, "r")

    # b
    bvals = read(file, "b")
    b = FEField(bvals, m.geom.g2)

    # ω, χ
    ωxvals = read(file, "ωx")
    ωyvals = read(file, "ωy")
    χxvals = read(file, "χx")
    χyvals = read(file, "χy")
    ωx = DGField(ωxvals, m.geom.g1)
    ωy = DGField(ωyvals, m.geom.g1)
    χx = DGField(χxvals, m.geom.g1)
    χy = DGField(χyvals, m.geom.g1)

    # Ψ
    Ψvals = read(file, "Ψ")
    Ψ = FEField(Ψvals, m.geom.g_sfc1)

    # time
    t = read(file, "t")

    close(file)

    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, t)
    return s
end