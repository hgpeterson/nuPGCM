################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState3D{IN<:Integer,F<:Field,FV<:AbstractVector}
    # buoyancy
	b::FV

    # vorticity
	ωx::F
	ωy::F

    # streamfunction
    χx::F
    χy::F

    # iteration
    i::IN
end

struct ModelSetup3D{FT<:AbstractFloat,F<:Field,G<:Grid,M<:AbstractMatrix,GV1<:AbstractVector,ZV<:AbstractVector,IV<:AbstractVector,CV<:AbstractVector,
                    GV2<:AbstractVector,DV<:AbstractVector,FV<:AbstractVector,FA<:Factorization,FTV<:AbstractVector}
    ε²::FT
    μ::FT
    ϱ::FT
    Δt::FT
    H::F
    Hx::F
    Hy::F
    f::F
    fy::F
    τx::F
    τy::F
    τx_x::F
    τx_y::F
    τy_x::F
    τy_y::F
    g_sfc::G
    g::G
    g_cols::GV1
    z_cols::ZV
    nzs::IV
    p_to_tri::CV
    b_cols::GV2
    Dxs::DV
    Dys::DV
    baroclinic_LHSs::FV
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

################################################################################
# Constructors for ModelSetup2DPG
################################################################################

function ModelSetup3D(ε², μ, ϱ, Δt, f, β, H::Function, τx::Function, τy::Function, g_sfc1, nσ=0, chebyshev=false)
    # second order surface mesh
    g_sfc1 = Grid(2, g_sfc1)

    # 3d mesh
    g1, g2, σ = generate_wedge_cols(g_sfc1, g_sfc2, nσ=nσ, chebyshev=chebyshev)

    # convert functions to fields
    H = FEField(H, g_sfc2)
    τx = FEField(τx, g_sfc2)
    τy = FEField(τy, g_sfc2)

    # TODO: take their gradients here

    if showplots
        quick_plot(H, L"H", "$out_folder/H.png")
        quick_plot(Hx, L"H_x", "$out_folder/Hx.png")
        quick_plot(Hy, L"H_y", "$out_folder/Hy.png")
        f_over_H = f/(H + FEField(1e-5, g_sfc))
        quick_plot(f_over_H, L"f/H", "$out_folder/f_over_H.png", vmax=6)
        curl = (τy_x - τx_y)*H - (τy*Hx - τx*Hy)
        quick_plot(curl, L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", "$out_folder/curl.png")
    end

    # derivative matrices
    Dxs = Vector{Vector{SparseMatrixCSC}}(undef, g_sfc.nt)
    Dys = Vector{Vector{SparseMatrixCSC}}(undef, g_sfc.nt)
    @showprogress "Saving derivative matrices..." for k=1:g_sfc.nt
        Dxs[k], Dys[k] = get_b_gradient_matrices(b_cols[k], g_cols[k], nzs[g_sfc.t[k, :]]) 
    end

    # baroclinc LHSs
    baroclinic_LHSs = [nzs[i] > 1 ? get_baroclinic_LHS(z_cols[i], ε², f(g_sfc.p[i, :])) : lu(sparse([1;;])) for i ∈ eachindex(z_cols)]

    # get transport ω and χ
    ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = get_transport_ω_and_χ(baroclinic_LHSs, g_sfc, p_to_tri, z_cols, nzs, H, ε², showplots=showplots)
    ωx_Ux_bot = FEField([ωx_Ux[p_to_tri[i][1]][1] for i=1:g_sfc.np], g_sfc)/H^2
    ωy_Ux_bot = FEField([ωy_Ux[p_to_tri[i][1]][1] for i=1:g_sfc.np], g_sfc)/H^2

    # bottom drag coefficients
    r_sym = ωy_Ux_bot/H
    r_asym = ωx_Ux_bot/H

    # barotropic LHS
    barotropic_LHS = get_barotropic_LHS(g_sfc, r_sym, r_asym, f, fy, H, Hx, Hy, ε²)

    # get ω_τ's
    ωx_τx, ωy_τx, χx_τx, χy_τx = get_wind_ω_and_χ(baroclinic_LHSs, g_sfc, p_to_tri, z_cols, nzs, H, ε², showplots=showplots)
    ωx_τx_bot = FEField([ωx_τx[p_to_tri[i][1]][1] for i=1:g_sfc.np], g_sfc)/H^2
    ωy_τx_bot = FEField([ωy_τx[p_to_tri[i][1]][1] for i=1:g_sfc.np], g_sfc)/H^2
    ωx_τy_bot = -ωy_τx_bot
    ωy_τy_bot = ωx_τx_bot
    ωx_τ_bot = (τx*ωx_τx_bot + τy*ωx_τy_bot)/H
    ωy_τ_bot = (τx*ωy_τx_bot + τy*ωy_τy_bot)/H
    if showplots
        quick_plot(ωx_τ_bot*H, L"\omega^x_\tau(-H)", "$out_folder/omegax_tau_bot.png")
        quick_plot(ωy_τ_bot*H, L"\omega^y_\tau(-H)", "$out_folder/omegay_tau_bot.png")
    end

    # barotropic RHS due to wind stress
    barotropic_RHS_τ = get_barotropic_RHS_τ(g_sfc, H, Hx, Hy, τx, τy, τx_y, τy_x, ωx_τ_bot, ωy_τ_bot, ε²)

    return ModelSetup3D(ε², μ, ϱ, Δt, H, Hx, Hy, f, fy, τx, τy, τx_x, τx_y, τy_x, τy_y, g_sfc, g, g_cols, z_cols, nzs, p_to_tri, b_cols, Dxs, Dys, 
                        baroclinic_LHSs, ωx_Ux, ωy_Ux, χx_Ux, χy_Ux, barotropic_LHS, ωx_τx, ωy_τx, χx_τx, χy_τx, barotropic_RHS_τ)
end