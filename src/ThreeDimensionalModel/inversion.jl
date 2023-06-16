function invert(m::ModelSetup3D, b; showplots=false)
    # unpack
    g_sfc = m.g_sfc
    g_cols = m.g_cols
    p_to_tri = m.p_to_tri
    H = m.H

    # get buoyancy ω and χ
    ωx_b, ωy_b, χx_b, χy_b = get_buoyancy_ω_and_χ(m, b, showplots=showplots)
    ωx_b_bot = DGField([ωx_b[k, i][1] for k=1:g_sfc.nt, i=1:g_sfc.nn], g_sfc)/H
    ωy_b_bot = DGField([ωy_b[k, i][1] for k=1:g_sfc.nt, i=1:g_sfc.nn], g_sfc)/H

    # solve barotropic
    barotropic_RHS_b = get_barotropic_RHS_b(m, b, ωx_b_bot, ωy_b_bot)
    Ψ = m.barotropic_LHS\(m.barotropic_RHS_τ + barotropic_RHS_b)
    Ψ = FEField(Ψ, g_sfc)
    if showplots
        quick_plot(Ψ, L"\Psi", "$out_folder/psi.png")
    end

    # take gradients to get Uˣ and Uʸ
    Ux, Uy = get_Ux_Uy(Ψ, showplots=showplots)

    # put them all together to get full ω's and χ's
    nzs = [size(z, 1) for z ∈ m.z_cols]
    ωx = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    ωy = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    χx = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    χy = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    for i=1:g_sfc.np
        for I ∈ p_to_tri[i]
            ωx[I] = ωx_b[I] + Ux[I[1]]*m.ωx_Ux[I]/m.H[i]^2 - Uy[I[1]]*m.ωy_Ux[I]/m.H[i]^2
            ωy[I] = ωy_b[I] + Ux[I[1]]*m.ωy_Ux[I]/m.H[i]^2 + Uy[I[1]]*m.ωx_Ux[I]/m.H[i]^2
            χx[I] = χx_b[I] + Ux[I[1]]*m.χx_Ux[I]/m.H[i]^2 - Uy[I[1]]*m.χy_Ux[I]/m.H[i]^2
            χy[I] = χy_b[I] + Ux[I[1]]*m.χy_Ux[I]/m.H[i]^2 + Uy[I[1]]*m.χx_Ux[I]/m.H[i]^2
        end
    end
    if showplots
        plot_ω_χ(ωx, ωy, χx, χy, g_cols)
    end

    # cg for now
    g = m.g
    ωx_cg = zeros(g.np)
    ωy_cg = zeros(g.np)
    χx_cg = zeros(g.np)
    χy_cg = zeros(g.np)
    n = 0
    for i=1:g_sfc.np
        nz = nzs[i]
        weight = size(p_to_tri[i], 1)
        for I ∈ p_to_tri[i]
            ωx_cg[n+1:n+nz] += ωx[I]/weight
            ωy_cg[n+1:n+nz] += ωy[I]/weight
            χx_cg[n+1:n+nz] += χx[I]/weight
            χy_cg[n+1:n+nz] += χy[I]/weight
        end
        n += nz
    end
    ωx = FEField(ωx_cg, g)
    ωy = FEField(ωy_cg, g)
    χx = FEField(χx_cg, g)
    χy = FEField(χy_cg, g)
    if showplots
        write_vtk(g, "$out_folder/omega_chi_cg.vtu", Dict("omega^x"=>ωx, "omega^y"=>ωy, "chi^x"=>χx, "chi^y"=>χy))
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