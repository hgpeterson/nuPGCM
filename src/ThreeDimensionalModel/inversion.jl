function invert(m::ModelSetup3D, b, γ; showplots=false, nonzero_b=true)
    # unpack
    g_sfc = m.g_sfc
    g_cols = m.g_cols
    p_to_tri = m.p_to_tri
    H = m.H

    # get buoyancy ω and χ
    if nonzero_b
        ωx_b, ωy_b, χx_b, χy_b = get_buoyancy_ω_and_χ(m, b, showplots=showplots)
        ωx_b_bot = DGField([ωx_b[k, i][1] for k=1:g_sfc.nt, i=1:g_sfc.nn], g_sfc)/H
        ωy_b_bot = DGField([ωy_b[k, i][1] for k=1:g_sfc.nt, i=1:g_sfc.nn], g_sfc)/H
    else
        ωx_b_bot = DGField(0, g_sfc)
        ωy_b_bot = DGField(0, g_sfc)
    end

    # solve barotropic
    barotropic_RHS_b = get_barotropic_RHS_b(m, γ, ωx_b_bot, ωy_b_bot)
    Ψ = m.barotropic_LHS\(m.barotropic_RHS_τ + barotropic_RHS_b)
    Ψ = FEField(Ψ, g_sfc)
    if showplots
        quick_plot(Ψ, L"\Psi", "$out_folder/psi.png")
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
    nzs = [size(z, 1) for z ∈ m.z_cols]
    ωx = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    ωy = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    χx = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    χy = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    for i=1:g_sfc.np
        for I ∈ p_to_tri[i]
            ωx[I] = ωx_b[I] + Ux_cg[i]*m.ωx_Ux[I]/m.H[i]^2 - Uy_cg[i]*m.ωy_Ux[I]/m.H[i]^2
            ωy[I] = ωy_b[I] + Ux_cg[i]*m.ωy_Ux[I]/m.H[i]^2 + Uy_cg[i]*m.ωx_Ux[I]/m.H[i]^2
            χx[I] = χx_b[I] + Ux_cg[i]*m.χx_Ux[I]/m.H[i]^2 - Uy_cg[i]*m.χy_Ux[I]/m.H[i]^2
            χy[I] = χy_b[I] + Ux_cg[i]*m.χy_Ux[I]/m.H[i]^2 + Uy_cg[i]*m.χx_Ux[I]/m.H[i]^2
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
        quick_plot(Ux, L"U^x", "$out_folder/Ux.png")
        quick_plot(Uy, L"U^y", "$out_folder/Uy.png")
    end
    return Ux, Uy
end