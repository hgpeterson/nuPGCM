function invert(m::ModelSetup3D, b, γ; showplots=false, nonzero_b=true)
    # unpack
    g_sfc = m.g_sfc
    g_cols = m.g_cols
    p_to_tri = m.p_to_tri
    H = m.H

    # get buoyancy ω and χ
    if nonzero_b
        ωx_b, ωy_b, χx_b, χy_b = get_buoyancy_ω_and_χ(m, b, showplots=showplots)
        ωx_b_bot = [ωx_b[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3]
        ωy_b_bot = [ωy_b[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3]
        ωx_b_bot = DGField(ωx_b_bot, g_sfc)/H
        ωy_b_bot = DGField(ωy_b_bot, g_sfc)/H
    else
        ωx_b_bot = DGField(0, g_sfc)
        ωy_b_bot = DGField(0, g_sfc)
    end

    # solve barotropic
    barotropic_RHS_b = get_barotropic_RHS_b(m, γ, ωx_b_bot, ωy_b_bot)
    Ψ = m.barotropic_LHS\(m.barotropic_RHS_τ + barotropic_RHS_b)
    Ψ = FEField(Ψ, g_sfc)
    if showplots
        quick_plot(Ψ, L"\Psi", "scratch/images/psi.png")
    end

    # take gradients to get Uˣ and Uʸ
    Ux, Uy = get_Ux_Uy(Ψ, showplots=showplots)
    # for now: convert to CG
    Ux_cg = zeros(g_sfc.np)
    Uy_cg = zeros(g_sfc.np)
    for i=1:g_sfc.np
        Ux_cg[i] = sum(Ux[I[1]] for I ∈ p_to_tri[i])/size(p_to_tri[i], 1)
        Uy_cg[i] = sum(Uy[I[1]] for I ∈ p_to_tri[i])/size(p_to_tri[i], 1)
    end

    # put them all together to get full ω's and χ's
    ωx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    ωy = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χy = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    for k=1:g_sfc.nt
        n = 0
        for i=1:3
            ig = g_sfc.t[k, i]
            nz = size(z_cols[ig], 1)
            i_col = n+1:n+nz 
            # ωx[k][i_col] = ωx_b[k][i_col] + Ux[k]*ωx_Ux[k][i_col]/Hfield[ig]^2 - Uy[k]*ωy_Ux[k][i_col]/Hfield[ig]^2
            # ωy[k][i_col] = ωy_b[k][i_col] + Ux[k]*ωy_Ux[k][i_col]/Hfield[ig]^2 + Uy[k]*ωx_Ux[k][i_col]/Hfield[ig]^2
            # χx[k][i_col] = χx_b[k][i_col] + Ux[k]*χx_Ux[k][i_col]/Hfield[ig]^2 - Uy[k]*χy_Ux[k][i_col]/Hfield[ig]^2
            # χy[k][i_col] = χy_b[k][i_col] + Ux[k]*χy_Ux[k][i_col]/Hfield[ig]^2 + Uy[k]*χx_Ux[k][i_col]/Hfield[ig]^2
            ωx[k][i_col] = ωx_b[k][i_col] + Ux_cg[ig]*m.ωx_Ux[k][i_col]/m.H[ig]^2 - Uy_cg[ig]*m.ωy_Ux[k][i_col]/m.H[ig]^2
            ωy[k][i_col] = ωy_b[k][i_col] + Ux_cg[ig]*m.ωy_Ux[k][i_col]/m.H[ig]^2 + Uy_cg[ig]*m.ωx_Ux[k][i_col]/m.H[ig]^2
            χx[k][i_col] = χx_b[k][i_col] + Ux_cg[ig]*m.χx_Ux[k][i_col]/m.H[ig]^2 - Uy_cg[ig]*m.χy_Ux[k][i_col]/m.H[ig]^2
            χy[k][i_col] = χy_b[k][i_col] + Ux_cg[ig]*m.χy_Ux[k][i_col]/m.H[ig]^2 + Uy_cg[ig]*m.χx_Ux[k][i_col]/m.H[ig]^2
            # x = g_sfc.p[ig, :]
            # ωx[k][i_col] .= Ux_cg[ig]#*x[1]*z_cols[ig]
            # ωy[k][i_col] .= Uy_cg[ig]#*x[1]*z_cols[ig]
            # χx[k][i_col] .= Ux_cg[ig]#*x[1]^2*z_cols[ig]
            # χy[k][i_col] .= Uy_cg[ig]#*x[1]^2*z_cols[ig]
            n += nz
        end
    end
    if showplots
        plot_ω_χ(ωx, ωy, χx, χy, g_cols)
    end

    return ωx, ωy, χx, χy, Ψ
end

function get_Ux_Uy(Ψ; showplots=false)
    g = Ψ.g
    Ux = FVField([-∂y(Ψ, [0, 0], k) for k=1:g.nt], g)
    Uy = FVField([+∂x(Ψ, [0, 0], k) for k=1:g.nt], g)
    if showplots
        quick_plot(Ux, L"U^x", "scratch/images/Ux.png")
        quick_plot(Uy, L"U^y", "scratch/images/Uy.png")
    end
    return Ux, Uy
end