function invert!(m::ModelSetup3D, b, ωx, ωy, χx, χy, Ψ; showplots=false)
    # unpack
    g_sfc1 = m.g_sfc1
    nσ = m.nσ
    ν_bot = m.ν_bot
    in_nodes1 = m.in_nodes1

    # get buoyancy ω and χ
    ωx_b, ωy_b, χx_b, χy_b = solve_baroclinic_buoyancy(m, b, showplots=showplots)
    νωx_b_bot = DGField([ν_bot[g_sfc1.t[k, i]]*ωx_b[k, i, 1] for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    νωy_b_bot = DGField([ν_bot[g_sfc1.t[k, i]]*ωy_b[k, i, 1] for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    # solve barotropic
    barotropic_RHS_b = build_barotropic_RHS_b(m, b, νωx_b_bot, νωy_b_bot, showplots=showplots)
    Ψ.values[:] = m.barotropic_LHS\(m.barotropic_RHS_τ + barotropic_RHS_b)
    if showplots
        quick_plot(Ψ, L"Barotropic streamfunction $\Psi$", "$out_folder/psi.png")
    end

    # take gradients to get Uˣ and Uʸ
    Ux, Uy = compute_U(Ψ, showplots=showplots)

    # put them all together to get full ω's and χ's
    for ig ∈ in_nodes1
        for I ∈ g_sfc1.p_to_t[ig]
            k = I[1]
            i = I[2]
            for j=1:nσ-1
                k_w = get_k_w(k, nσ, j)
                ωx.values[k_w, i] = ωx_b[k, i, j] + Ux[k]*m.ωx_Ux[ig, j]/m.H[ig]^2 - Uy[k]*m.ωy_Ux[ig, j]/m.H[ig]^2 #FIXME: add τ's
                ωy.values[k_w, i] = ωy_b[k, i, j] + Ux[k]*m.ωy_Ux[ig, j]/m.H[ig]^2 + Uy[k]*m.ωx_Ux[ig, j]/m.H[ig]^2
                χx.values[k_w, i] = χx_b[k, i, j] + Ux[k]*m.χx_Ux[ig, j]/m.H[ig]^2 - Uy[k]*m.χy_Ux[ig, j]/m.H[ig]^2
                χy.values[k_w, i] = χy_b[k, i, j] + Ux[k]*m.χy_Ux[ig, j]/m.H[ig]^2 + Uy[k]*m.χx_Ux[ig, j]/m.H[ig]^2
                ωx.values[k_w, i+3] = ωx_b[k, i, j+1] + Ux[k]*m.ωx_Ux[ig, j+1]/m.H[ig]^2 - Uy[k]*m.ωy_Ux[ig, j+1]/m.H[ig]^2 
                ωy.values[k_w, i+3] = ωy_b[k, i, j+1] + Ux[k]*m.ωy_Ux[ig, j+1]/m.H[ig]^2 + Uy[k]*m.ωx_Ux[ig, j+1]/m.H[ig]^2
                χx.values[k_w, i+3] = χx_b[k, i, j+1] + Ux[k]*m.χx_Ux[ig, j+1]/m.H[ig]^2 - Uy[k]*m.χy_Ux[ig, j+1]/m.H[ig]^2
                χy.values[k_w, i+3] = χy_b[k, i, j+1] + Ux[k]*m.χy_Ux[ig, j+1]/m.H[ig]^2 + Uy[k]*m.χx_Ux[ig, j+1]/m.H[ig]^2
            end
        end
    end
    if showplots
        # save .vtu
        plot_ω_χ(m, ωx, ωy, χx, χy)

        # profile and slice plots
        plot_profiles(m, b, ωx, ωy, χx, χy,  0.5, 0.0, fname="$out_folder/profiles_x=+0.5_y=0.0.png")
        plot_profiles(m, b, ωx, ωy, χx, χy, -0.5, 0.0, fname="$out_folder/profiles_x=-0.5_y=0.0.png")
        plot_profiles(m, b, ωx, ωy, χx, χy, 0.0,  0.5, fname="$out_folder/profiles_x=0.0_y=+0.5.png")
        plot_profiles(m, b, ωx, ωy, χx, χy, 0.0, -0.5, fname="$out_folder/profiles_x=0.0_y=-0.5.png")
        plot_xslice(m, b, χx, 0.0, fname="$out_folder/xslice_chix.png", cb_label=L"Streamfunction $\chi^x$")
        plot_xslice(m, b, χy, 0.0, fname="$out_folder/xslice_chiy.png", cb_label=L"Streamfunction $\chi^y$")
        plot_yslice(m, b, χx, 0.0, fname="$out_folder/yslice_chix.png", cb_label=L"Streamfunction $\chi^x$")
        plot_yslice(m, b, χy, 0.0, fname="$out_folder/yslice_chiy.png", cb_label=L"Streamfunction $\chi^y$")
        plot_xslice(m, b, ωx, 0.0, fname="$out_folder/xslice_omegax.png", cb_label=L"Vorticity $\omega^x$")
        plot_xslice(m, b, ωy, 0.0, fname="$out_folder/xslice_omegay.png", cb_label=L"Vorticity $\omega^y$")
        plot_yslice(m, b, ωx, 0.0, fname="$out_folder/yslice_omegax.png", cb_label=L"Vorticity $\omega^x$")
        plot_yslice(m, b, ωy, 0.0, fname="$out_folder/yslice_omegay.png", cb_label=L"Vorticity $\omega^y$")
    end

    return ωx, ωy, χx, χy, Ψ
end
function invert(m::ModelSetup3D, b; kwargs...)
    ωx = DGField(0, m.g1)
    ωy = DGField(0, m.g1)
    χx = DGField(0, m.g1)
    χy = DGField(0, m.g1)
    Ψ = FEField(0, m.g_sfc1)
    return invert!(m, b, ωx, ωy, χx, χy, Ψ; kwargs...)
end
function invert!(m::ModelSetup3D, s::ModelState3D; kwargs...)
    invert!(m, s.b, s.ωx, s.ωy, s.χx, s.χy, s.Ψ; kwargs...)
    return s
end

function compute_U(Ψ; showplots=false)
    g = Ψ.g
    Ux = FVField([-∂y(Ψ, [0, 0], k) for k=1:g.nt], g)
    Uy = FVField([+∂x(Ψ, [0, 0], k) for k=1:g.nt], g)
    # # set to zero on bdy
    # for i ∈ g.e["bdy"]
    #     for I ∈ g.p_to_t[i]
    #         Ux.values[I[1]] = 0
    #         Uy.values[I[1]] = 0
    #     end
    # end
    if showplots
        quick_plot(Ux, L"U^x", "$out_folder/Ux.png")
        quick_plot(Uy, L"U^y", "$out_folder/Uy.png")
    end
    return Ux, Uy
end