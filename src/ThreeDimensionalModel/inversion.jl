struct InversionComponents{D, FAV, M, FA, FTV}
    Dx::D
    Dy::D
    baroclinic_LHSs::FAV
    ωx_Ux::M
    ωy_Ux::M
    χx_Ux::M
    χy_Ux::M
    barotropic_LHS::FA
    ωx_τx::M
    ωy_τx::M
    χx_τx::M
    χy_τx::M
    barotropic_RHS_τ::FTV
end

"""
    inversion = InversionComponents(params::Params, geom::Geometry, forcing::Forcing)
"""
function InversionComponents(params::Params, geom::Geometry, forcing::Forcing)
    # unpack
    g_sfc1 = geom.g_sfc1
    ν_bot = forcing.ν_bot
    τx = forcing.τx
    τy = forcing.τy
    τx1 = FEField(τx[1:g_sfc1.np], g_sfc1)
    τy1 = FEField(τy[1:g_sfc1.np], g_sfc1)

    # derivative matrices
    Dx, Dy = build_b_gradient_matrices(geom) 
    
    # baroclinc LHS for each node column on first order grid
    baroclinic_LHSs = build_baroclinic_LHSs(params, geom, forcing)

    # transport ω and χ
    ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = solve_baroclinic_transport(geom, baroclinic_LHSs, showplots=true)
    νωx_Ux_bot = ν_bot*FEField(ωx_Ux[:, 1], g_sfc1)
    νωy_Ux_bot = ν_bot*FEField(ωy_Ux[:, 1], g_sfc1)

    # barotropic LHS
    barotropic_LHS = build_barotropic_LHS(params, geom, νωx_Ux_bot, νωy_Ux_bot)

    # wind stress ω and χ
    ωx_τx, ωy_τx, χx_τx, χy_τx = solve_baroclinic_wind(geom, params, baroclinic_LHSs, showplots=true)
    νωx_τx_bot = ν_bot*FEField(ωx_τx[:, 1], g_sfc1)
    νωy_τx_bot = ν_bot*FEField(ωy_τx[:, 1], g_sfc1)
    νωx_τ_bot = τx1*νωx_τx_bot - τy1*νωy_τx_bot
    νωy_τ_bot = τx1*νωy_τx_bot + τy1*νωx_τx_bot
    quick_plot(νωx_τ_bot, cb_label=L"\nu\omega^x_\tau|_{-H}", filename="$out_folder/nu_omegax_tau_bot.png")
    quick_plot(νωy_τ_bot, cb_label=L"\nu\omega^y_\tau|_{-H}", filename="$out_folder/nu_omegay_tau_bot.png")

    # barotropic RHS due to wind stress
    barotropic_RHS_τ = build_barotropic_RHS_τ(params, geom, forcing, νωx_τ_bot, νωy_τ_bot)

    return InversionComponents(Dx, Dy, baroclinic_LHSs, ωx_Ux, ωy_Ux, χx_Ux, χy_Ux, barotropic_LHS, ωx_τx, ωy_τx, χx_τx, χy_τx, barotropic_RHS_τ)
end

"""
    s = invert!(m::ModelSetup3D, s::ModelState3D; showplots=false)
"""
function invert!(m::ModelSetup3D, s::ModelState3D; showplots=false)
    # unpack
    g_sfc1 = m.geom.g_sfc1
    nσ = m.geom.nσ
    in_nodes1 = m.geom.in_nodes1
    H = m.geom.H
    ν_bot = m.forcing.ν_bot
    barotropic_LHS = m.inversion.barotropic_LHS
    barotropic_RHS_τ = m.inversion.barotropic_RHS_τ
    ωx_Ux = m.inversion.ωx_Ux
    ωy_Ux = m.inversion.ωy_Ux
    χx_Ux = m.inversion.χx_Ux
    χy_Ux = m.inversion.χy_Ux
    b = s.b
    ωx = s.ωx
    ωy = s.ωy
    χx = s.χx
    χy = s.χy
    Ψ = s.Ψ

    # get buoyancy ω and χ
    # @time "\tsolve_baroclinic_buoyancy" ωx_b, ωy_b, χx_b, χy_b = solve_baroclinic_buoyancy(m, b, showplots=showplots)
    ωx_b, ωy_b, χx_b, χy_b = solve_baroclinic_buoyancy(m, b, showplots=showplots)
    νωx_b_bot = DGField([ν_bot[g_sfc1.t[k, i]]*ωx_b[k, i, 1] for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    νωy_b_bot = DGField([ν_bot[g_sfc1.t[k, i]]*ωy_b[k, i, 1] for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    # solve barotropic
    # @time "\tbuild_barotropic_RHS_b" barotropic_RHS_b = build_barotropic_RHS_b(m, b, νωx_b_bot, νωy_b_bot, showplots=showplots)
    # @time "\tinvert" Ψ.values[:] = barotropic_LHS\(barotropic_RHS_τ + barotropic_RHS_b)
    barotropic_RHS_b = build_barotropic_RHS_b(m, b, νωx_b_bot, νωy_b_bot, showplots=showplots)
    Ψ.values[:] = barotropic_LHS\(barotropic_RHS_τ + barotropic_RHS_b)

    # take gradients to get Uˣ and Uʸ
    # @time "\tcompute_U" Ux, Uy = compute_U(Ψ, showplots=showplots)
    Ux, Uy = compute_U(Ψ)

    # put them all together to get full ω's and χ's
    # @time "\tsum" begin
    for ig ∈ in_nodes1
        for I ∈ g_sfc1.p_to_t[ig]
            k = I[1]
            i = I[2]
            for j=1:nσ-1
                k_w = get_k_w(k, nσ, j)
                ωx.values[k_w, i] = ωx_b[k, i, j] + Ux[k]*ωx_Ux[ig, j]/H[ig]^2 - Uy[k]*ωy_Ux[ig, j]/H[ig]^2 #FIXME: add τ's
                ωy.values[k_w, i] = ωy_b[k, i, j] + Ux[k]*ωy_Ux[ig, j]/H[ig]^2 + Uy[k]*ωx_Ux[ig, j]/H[ig]^2
                χx.values[k_w, i] = χx_b[k, i, j] + Ux[k]*χx_Ux[ig, j]/H[ig]^2 - Uy[k]*χy_Ux[ig, j]/H[ig]^2
                χy.values[k_w, i] = χy_b[k, i, j] + Ux[k]*χy_Ux[ig, j]/H[ig]^2 + Uy[k]*χx_Ux[ig, j]/H[ig]^2
                ωx.values[k_w, i+3] = ωx_b[k, i, j+1] + Ux[k]*ωx_Ux[ig, j+1]/H[ig]^2 - Uy[k]*ωy_Ux[ig, j+1]/H[ig]^2 
                ωy.values[k_w, i+3] = ωy_b[k, i, j+1] + Ux[k]*ωy_Ux[ig, j+1]/H[ig]^2 + Uy[k]*ωx_Ux[ig, j+1]/H[ig]^2
                χx.values[k_w, i+3] = χx_b[k, i, j+1] + Ux[k]*χx_Ux[ig, j+1]/H[ig]^2 - Uy[k]*χy_Ux[ig, j+1]/H[ig]^2
                χy.values[k_w, i+3] = χy_b[k, i, j+1] + Ux[k]*χy_Ux[ig, j+1]/H[ig]^2 + Uy[k]*χx_Ux[ig, j+1]/H[ig]^2
            end
        end
    end
    # end

    if showplots
        title = latexstring(L"$t = $", @sprintf("%.3f", m.params.Δt*s.i[1]))
        quick_plot(Ψ,  cb_label=L"Barotropic streamfunction $\Psi$", title=title, filename="$out_folder/psi.png")
        quick_plot(Ux, cb_label=L"U^x", title=title, filename="$out_folder/Ux.png")
        quick_plot(Uy, cb_label=L"U^y", title=title, filename="$out_folder/Uy.png")

        # save .vtu
        plot_ω_χ(m, ωx, ωy, χx, χy)

        # profile and slice plots
        # plot_profiles(m, b, ωx, ωy, χx, χy,  0.5, 0.0, "$out_folder/profiles_x=+0.5_y=0.0.png")
        # plot_profiles(m, b, ωx, ωy, χx, χy, -0.5, 0.0, "$out_folder/profiles_x=-0.5_y=0.0.png")
        # plot_profiles(m, b, ωx, ωy, χx, χy, 0.0,  0.5, "$out_folder/profiles_x=0.0_y=+0.5.png")
        # plot_profiles(m, b, ωx, ωy, χx, χy, 0.0, -0.5, "$out_folder/profiles_x=0.0_y=-0.5.png")
        plot_xslice(m, b, χx, 0.0, L"Streamfunction $\chi^x$", "$out_folder/xslice_chix.png")
        plot_xslice(m, b, χy, 0.0, L"Streamfunction $\chi^y$", "$out_folder/xslice_chiy.png")
        plot_yslice(m, b, χx, 0.0, L"Streamfunction $\chi^x$", "$out_folder/yslice_chix.png")
        plot_yslice(m, b, χy, 0.0, L"Streamfunction $\chi^y$", "$out_folder/yslice_chiy.png")
        plot_xslice(m, b, ωx, 0.0, L"Vorticity $\omega^x$", "$out_folder/xslice_omegax.png")
        plot_xslice(m, b, ωy, 0.0, L"Vorticity $\omega^y$", "$out_folder/xslice_omegay.png")
        plot_yslice(m, b, ωx, 0.0, L"Vorticity $\omega^x$", "$out_folder/yslice_omegax.png")
        plot_yslice(m, b, ωy, 0.0, L"Vorticity $\omega^y$", "$out_folder/yslice_omegay.png")
    end

    return s
end

"""
    Ux, Uy = compute_U(Ψ)
"""
function compute_U(Ψ)
    g = Ψ.g
    Ux = FVField([-∂η(Ψ, [0, 0], k) for k=1:g.nt], g)
    Uy = FVField([+∂ξ(Ψ, [0, 0], k) for k=1:g.nt], g)
    return Ux, Uy
end

function initial_state(m::ModelSetup3D, b; showplots=false)
    ωx = DGField(0, m.geom.g1)
    ωy = DGField(0, m.geom.g1)
    χx = DGField(0, m.geom.g1)
    χy = DGField(0, m.geom.g1)
    Ψ = FEField(0, m.geom.g_sfc1)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, [0])
    return invert!(m, s; showplots)
end