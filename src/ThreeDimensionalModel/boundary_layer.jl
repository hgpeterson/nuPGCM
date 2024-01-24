function invert_BL(m::ModelSetup3D, s::ModelState3D)
    # unpack
    g_col = m.geom.g_col
    g_sfc1 = m.geom.g_sfc1
    nσ = m.geom.nσ
    in_nodes1 = m.geom.in_nodes1
    H = m.geom.H
    g1 = m.geom.g1
    σ = m.geom.σ
    coast_mask = m.geom.coast_mask
    g_sfc1_to_g1_map = m.geom.g_sfc1_to_g1_map
    M_bc = m.inversion.M_bc
    Dx = m.inversion.Dx
    Dy = m.inversion.Dy
    f = m.params.f
    β = m.params.β
    ε² = m.params.ε²
    ν = m.forcing.ν

    # 1D mass matrix for interior ω
    M = mass_matrix(g_col)

    # build BL LHSs
    baroclinic_LHSs = build_baroclinic_LHSs(m.params, m.geom, m.forcing; bl=true)

    # compute gradients
    bx = reshape(Dx*s.b.values, (g_sfc1.nt, g_sfc1.nn, 2nσ-2))
    by = reshape(Dy*s.b.values, (g_sfc1.nt, g_sfc1.nn, 2nσ-2))

    # pre-allocate
    ωx_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)
    ωy_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)
    χx_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)
    χy_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)

    # q = √f/2ν
    q = FEField(x->sqrt((f + β*x[2])/2), g_sfc1)/FEField(sqrt.(ν[g1.e["bot"]]), g_sfc1)
    ε = √ε²

    # compute and store
    for i ∈ eachindex(in_nodes1) # H = 0 solution: all zeros
        ig = in_nodes1[i]
        for I ∈ g_sfc1.p_to_t[ig]
            # interior ω
            y = g_sfc1.p[ig, 2]
            ωx_b[I, :] += -1/(f + β*y)*M\(M_bc*bx[I, :])
            ωy_b[I, :] += -1/(f + β*y)*M\(M_bc*by[I, :])

            # interior O(1) χ
            r = build_baroclinic_RHS(g_col, M_bc, bx[I, :], by[I, :], 0, 0, 0, 0; bl=true)
            sol = baroclinic_LHSs[i]\r
            χx_b[I, :] += sol[0*nσ+1:1*nσ]
            χy_b[I, :] += sol[1*nσ+1:2*nσ]

            # interior O(ε) χ
            dχxdz_bot = ∂(FEField(χx_b[I, :], g_col), -1, 1)/H[ig]
            dχydz_bot = ∂(FEField(χy_b[I, :], g_col), -1, 1)/H[ig]
            q0 = q[ig]
            c1 = -q0*(dχxdz_bot - dχydz_bot)
            c2 = -q0*(dχxdz_bot + dχydz_bot)
            χx_b[I, :] += -ε*c2*σ/(2q0^2)
            χy_b[I, :] += +ε*c1*σ/(2q0^2)

            # BL correction
            z_b = (σ .+ 1)*H[ig]/ε
            ωx_b[I, :] += @. 1/ε*exp(-q0*z_b)*(c1*cos(q0*z_b) + c2*sin(q0*z_b))
            ωy_b[I, :] += @. 1/ε*exp(-q0*z_b)*(c2*cos(q0*z_b) - c1*sin(q0*z_b))
            χx_b[I, :] += @. ε*exp(-q0*z_b)*(c1*sin(q0*z_b) - c2*cos(q0*z_b))/(2q0^2)
            χy_b[I, :] += @. ε*exp(-q0*z_b)*(c1*cos(q0*z_b) + c2*sin(q0*z_b))/(2q0^2)
        end
    end

    ωx_b_bot = DGField(ωx_b[:, :, 1], g_sfc1)
    ωy_b_bot = DGField(ωy_b[:, :, 1], g_sfc1)
    quick_plot(ωx_b_bot, cb_label=L"\omega^x_b(-H)", filename="$out_folder/images/omegax_b_bot_BL.png")
    quick_plot(ωy_b_bot, cb_label=L"\omega^y_b(-H)", filename="$out_folder/images/omegay_b_bot_BL.png")

    ωx_b0, ωy_b0, χx_b0, χy_b0 = solve_baroclinic_buoyancy(m, s.b)
    ωx_b0_bot = DGField(ωx_b0[:, :, 1], g_sfc1)
    ωy_b0_bot = DGField(ωy_b0[:, :, 1], g_sfc1)
    quick_plot(abs(ωx_b_bot - ωx_b0_bot), cb_label=L"$\omega^x_b(-H)$ error", filename="$out_folder/images/omegax_b_bot_BL_err.png")
    quick_plot(abs(ωy_b_bot - ωy_b0_bot), cb_label=L"$\omega^y_b(-H)$ error", filename="$out_folder/images/omegay_b_bot_BL_err.png")

    ωx_b = DGField((coast_mask .* ωx_b)[g_sfc1_to_g1_map], g1)
    ωy_b = DGField((coast_mask .* ωy_b)[g_sfc1_to_g1_map], g1)
    χx_b = DGField((coast_mask .* χx_b)[g_sfc1_to_g1_map], g1)
    χy_b = DGField((coast_mask .* χy_b)[g_sfc1_to_g1_map], g1)
    ωx_b0 = DGField((coast_mask .* ωx_b0)[g_sfc1_to_g1_map], g1)
    ωy_b0 = DGField((coast_mask .* ωy_b0)[g_sfc1_to_g1_map], g1)
    χx_b0 = DGField((coast_mask .* χx_b0)[g_sfc1_to_g1_map], g1)
    χy_b0 = DGField((coast_mask .* χy_b0)[g_sfc1_to_g1_map], g1)

    plot_xslice(m, s.b, ωx_b, 0, L"$\omega^x_b$", "$out_folder/images/omegax_b_slice_BL.png")
    plot_xslice(m, s.b, ωx_b0, 0, L"$\omega^x_b$", "$out_folder/images/omegax_b_slice.png")
    plot_xslice(m, s.b, ωy_b, 0, L"$\omega^x_b$", "$out_folder/images/omegay_b_slice_BL.png")
    plot_xslice(m, s.b, ωy_b0, 0, L"$\omega^x_b$", "$out_folder/images/omegay_b_slice.png")
    # fig, ax = plt.subplots(2, 2, figsize=(3.2, 5.2))
    # ax[1, 1].plot(ωx, z, label=L"\omega^x")
    # ax[1, 1].plot(ωy, z, label=L"\omega^y")
    # ax[1, 1].plot(ωx_BL, z, "k--", lw=0.5)
    # ax[1, 1].plot(ωy_BL, z, "k--", lw=0.5)
    # ax[1, 2].plot(χx, z, label=L"\chi^x")
    # ax[1, 2].plot(χy, z, label=L"\chi^y")
    # ax[1, 2].plot(χx_BL, z, "k--", lw=0.5)
    # ax[1, 2].plot(χy_BL, z, "k--", lw=0.5)
    # ax[2, 1].plot(ωx, z, label=L"\omega^x")
    # ax[2, 1].plot(ωy, z, label=L"\omega^y")
    # ax[2, 1].plot(ωx_BL, z, "k--", lw=0.5)
    # ax[2, 1].plot(ωy_BL, z, "k--", lw=0.5)
    # ax[2, 2].plot(χx, z, label=L"\chi^x")
    # ax[2, 2].plot(χy, z, label=L"\chi^y")
    # ax[2, 2].plot(χx_BL, z, "k--", lw=0.5)
    # ax[2, 2].plot(χy_BL, z, "k--", lw=0.5)
    # ax[1, 1].set_ylabel(L"z")
    # ax[2, 1].set_ylabel(L"z")
    # ax[2, 1].set_xlabel(L"\omega")
    # ax[2, 2].set_xlabel(L"\chi")
    # ax[1, 1].legend()
    # ax[1, 2].legend()
    # # ax[2, 1].set_xlim(-2/ε, 2/ε)
    # ax[2, 1].set_ylim(-H, -H + 5*ε/q)
    # # ax[2, 2].set_xlim(-2*ε, 2*ε)
    # ax[2, 2].set_ylim(-H, -H + 5*ε/q)
    # ax[1, 2].set_yticklabels([])
    # ax[2, 2].set_yticklabels([])
    # savefig("$out_folder/images/omega_chi_BL.png")
    # println("$out_folder/images/omega_chi_BL.png")
    # plt.close()
end

function test_1d()
    # params
    ε² = 1e-4
    ε = sqrt(ε²)
    f = 1 + 0.95*0.0

    # grid
    nσ = 2^8
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2
    H = 0.5
    z = H*σ
    p = σ
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g = Grid(Line(order=1), p, t, e)

    # forcing
    ν = @. 1e-2 + exp(-H*(σ + 1)/0.1)
    z_dg = zeros(2nσ-2)
    for i ∈ 1:nσ-1
        z_dg[2i-1] = z[i]
        z_dg[2i]   = z[i+1]
    end
    bx = @. z_dg*exp(-(z_dg + H)/(0.1*H))
    by = @. exp(-(z_dg + H)/(0.1*H))
    # bx = z_dg
    # by = ones(2nσ-2)
    Ux = 0
    Uy = 0
    τx = 0
    τy = 0

    # numerical sol
    A = build_baroclinic_LHS(g, ν, H, ε², f)
    r = build_baroclinic_RHS(g, bx, by, Ux, Uy, τx, τy)
    sol = A\r
    ωx = sol[0nσ+1:1nσ]
    ωy = sol[1nσ+1:2nσ]
    χx = sol[2nσ+1:3nσ]
    χy = sol[3nσ+1:4nσ]

    # BL sol
    q = sqrt(f/2/ν[1])
    z_b = (z .+ H)/ε

    # # transport
    # c1 = -q/H
    # c2 = +q/H
    # χx_I0 = 0
    # χy_I0 = @. -(z + H)/H
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))
    # χy_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωx_BL = 1/ε*ωx_B1
    # ωy_BL = 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ ε*(χx_I1 .+ χx_B1)
    # χy_BL = χy_I0 .+ ε*(χy_I1 .+ χy_B1)

    # # wind
    # c1 = c2 = -1/(2*H*q)
    # χx_I0 = @. (z + H)/(2*H*q^2)
    # χy_I0 = 0
    # ωx0_B0 = @. -exp(q*z_s)*sin(q*z_s)
    # ωy0_B0 = @. exp(q*z_s)*cos(q*z_s)
    # χx0_B0 = @. -1/(2*q^2)*exp(q*z_s)*cos(q*z_s)
    # χy0_B0 = @. -1/(2*q^2)*exp(q*z_s)*sin(q*z_s)
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))
    # χy_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωx_BL = 1/ε²*ωx0_B0 .+ 1/ε*ωx_B1
    # ωy_BL = 1/ε²*ωy0_B0 .+ 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ χx0_B0 .+ ε*(χx_I1 .+ χx_B1)
    # χy_BL = χy_I0 .+ χy0_B0 .+ ε*(χy_I1 .+ χy_B1)

    # buoyancy
    A = build_baroclinic_LHS(g, ν, H, ε², f; bl=true)
    r = build_baroclinic_RHS(g, bx, by, Ux, Uy, τx, τy; bl=true)
    sol = A\r
    M = mass_matrix(g)
    M_bc = build_M_bc(g)
    ωx_I0 = -1/f*M\(M_bc*bx)
    ωy_I0 = -1/f*M\(M_bc*by)
    χx_I0 = sol[0nσ+1:1nσ]
    χy_I0 = sol[1nσ+1:2nσ]
    # dχxdz_bot = ∂(FEField(χx_I0, g), -1, 1)/H
    # dχydz_bot = ∂(FEField(χy_I0, g), -1, 1)/H
    # println(dχxdz_bot)
    # println(dχydz_bot)
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    dχxdz_bot = dot(fd_z, χx_I0[1:3])
    dχydz_bot = dot(fd_z, χy_I0[1:3])
    # println(dot(fd_z, χx_I0[1:3]))
    # println(dot(fd_z, χy_I0[1:3]))
    c1 = -q*(dχxdz_bot - dχydz_bot)
    c2 = -q*(dχxdz_bot + dχydz_bot)
    χx_I1 = -c2*z/(2q^2*H)
    χy_I1 = +c1*z/(2q^2*H)
    ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    χx_B1 = @. exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))/(2q^2)
    χy_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))/(2q^2)
    ωx_BL = ωx_I0 + 1/ε*(ωx_B1)
    ωy_BL = ωy_I0 + 1/ε*(ωy_B1)
    χx_BL = χx_I0 + ε*(χx_I1 + χx_B1)
    χy_BL = χy_I0 + ε*(χy_I1 + χy_B1)
    # ωx_I0 = -bx/y
    # ωy_I0 = -by/y
    # χx_I0 = @. (z^3 - z)/6 # bx = z
    # χy_I0 = @. (z^2 + z)/2 # by = 1
    # c1 = -ωx_I0[nz]
    # c2 = ωy_I0[nz]
    # ωx0_B0 = @. exp(q*z_s)*(c1*cos(q*z_s) + c2*sin(q*z_s))
    # ωy0_B0 = @. exp(q*z_s)*(c1*sin(q*z_s) - c2*cos(q*z_s))
    # χx0_B2 = @. exp(q*z_s)*(c2*cos(q*z_s) - c1*sin(q*z_s))/(2q^2)
    # χy0_B2 = @. exp(q*z_s)*(c1*sin(q*z_s) - c2*cos(q*z_s))/(2q^2)
    # c1 = -5q/6 # bx = z
    # c2 = q/6 # by = 1
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))/(2q^2)
    # χy_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))/(2q^2)
    # ωx_BL = ωx_I0 .+ ωx0_B0 .+ 1/ε*ωx_B1
    # ωy_BL = ωy_I0 .+ ωy0_B0 .+ 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ ε*(χx_I1 .+ χx_B1) .+ ε²*χx0_B2
    # χy_BL = χy_I0 .+ ε*(χy_I1 .+ χy_B1) .+ ε²*χy0_B2

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(3.2, 5.2))
    ax[1, 1].plot(ωx, z, label=L"\omega^x")
    ax[1, 1].plot(ωy, z, label=L"\omega^y")
    ax[1, 1].plot(ωx_BL, z, "k--", lw=0.5)
    ax[1, 1].plot(ωy_BL, z, "k--", lw=0.5)
    ax[1, 2].plot(χx, z, label=L"\chi^x")
    ax[1, 2].plot(χy, z, label=L"\chi^y")
    ax[1, 2].plot(χx_BL, z, "k--", lw=0.5)
    ax[1, 2].plot(χy_BL, z, "k--", lw=0.5)
    ax[2, 1].plot(ωx, z, label=L"\omega^x")
    ax[2, 1].plot(ωy, z, label=L"\omega^y")
    ax[2, 1].plot(ωx_BL, z, "k--", lw=0.5)
    ax[2, 1].plot(ωy_BL, z, "k--", lw=0.5)
    ax[2, 2].plot(χx, z, label=L"\chi^x")
    ax[2, 2].plot(χy, z, label=L"\chi^y")
    ax[2, 2].plot(χx_BL, z, "k--", lw=0.5)
    ax[2, 2].plot(χy_BL, z, "k--", lw=0.5)
    ax[1, 1].set_ylabel(L"z")
    ax[2, 1].set_ylabel(L"z")
    ax[2, 1].set_xlabel(L"\omega")
    ax[2, 2].set_xlabel(L"\chi")
    ax[1, 1].legend()
    ax[1, 2].legend()
    # ax[2, 1].set_xlim(-2/ε, 2/ε)
    ax[2, 1].set_ylim(-H, -H + 5*ε/q)
    # ax[2, 2].set_xlim(-2*ε, 2*ε)
    ax[2, 2].set_ylim(-H, -H + 5*ε/q)
    ax[1, 2].set_yticklabels([])
    ax[2, 2].set_yticklabels([])
    savefig("$out_folder/images/omega_chi_BL.png")
    println("$out_folder/images/omega_chi_BL.png")
    plt.close()
end

function test_2d()
    # params
    ε² = 1e-3
    ε = sqrt(ε²)
    f = 1
    β = 0.95
    ν_bot = 1.01

    # grid
    g = Grid(Triangle(order=2), "$(@__DIR__)/../../meshes/circle/mesh5.h5")

    # functions on grid
    H = FEField(x->1 - x[1]^2 - x[2]^2, g)
    q = FEField(x->√((f + β*x[2])/(2*ν_bot)), g)
    ωx_Ux_bot = -H*q/ε
    ωy_Ux_bot = H*q/ε

    # plot
    quick_plot(ωx_Ux_bot, cb_label=L"\omega^x_{U^x}(-H)", filename="$out_folder/images/omegax_Ux_bot_BL.png")
    quick_plot(ωy_Ux_bot, cb_label=L"\omega^y_{U^x}(-H)", filename="$out_folder/images/omegay_Ux_bot_BL.png")
end