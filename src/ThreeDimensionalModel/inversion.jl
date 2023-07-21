function invert!(m::ModelSetup3D, b, ωx, ωy, χx, χy, Ψ; showplots=false)
    # unpack
    g_sfc1 = m.g_sfc1
    nσ = m.nσ
    H = m.H

    # get buoyancy ω and χ
    ωx_b, ωy_b, χx_b, χy_b = get_buoyancy_ω_and_χ(m, b, showplots=showplots)
    ωx_b_bot = DGField([ωx_b[k, i, 1]/H[g_sfc1.t[k, i]] for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    ωy_b_bot = DGField([ωy_b[k, i, 1]/H[g_sfc1.t[k, i]] for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    # solve barotropic
    barotropic_RHS_b = get_barotropic_RHS_b(m, b, ωx_b_bot, ωy_b_bot, showplots=showplots)
    Ψ.values[:] = m.barotropic_LHS\(m.barotropic_RHS_τ + barotropic_RHS_b)
    if showplots
        quick_plot(Ψ, L"\Psi", "$out_folder/psi.png")
    end

    # take gradients to get Uˣ and Uʸ
    Ux, Uy = get_Ux_Uy(Ψ, showplots=showplots)

    # put them all together to get full ω's and χ's
    for k=1:g_sfc1.nt
        for i=1:g_sfc1.nn
            ig = g_sfc1.t[k, i]
            for j=1:nσ-1
                k_w = (k - 1)*(nσ - 1) + j
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
        plot_ω_χ(m, ωx, ωy, χx, χy)
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

function get_u(m::ModelSetup3D, s::ModelState3D; showplots=false)
    ux = FVField([-∂z(s.χy, [0, 0, 0], k) for k=1:m.g.nt], m.g)
    uy = FVField([+∂z(s.χx, [0, 0, 0], k) for k=1:m.g.nt], m.g)
    uz = FVField([∂x(s.χy, [0, 0, 0], k) - ∂y(s.χx, [0, 0, 0], k) for k=1:m.g.nt], m.g)
    if showplots
        cell_type = VTKCellTypes.VTK_TETRA
        cells = [MeshCell(cell_type, m.g.t[i, :]) for i ∈ axes(m.g.t, 1)]
        vtk_grid("$out_folder/u.vtu", m.g.p', cells) do vtk
            vtk["ux"] = ux.values
            vtk["uy"] = uy.values
            vtk["uz"] = uz.values
        end
        println("$out_folder/u.vtu")
    end
    return ux, uy, uz
end