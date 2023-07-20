################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState3D{I<:Integer,F1<:AbstractField,F2<:AbstractField}
    # buoyancy
	b::F2

    # vorticity
	ωx::F1
	ωy::F1

    # streamfunction
    χx::F1
    χy::F1

    # iteration
    i::I
end

struct ModelSetup3D{FT<:AbstractFloat,F1<:AbstractField,F2<:AbstractField,GS1<:Grid,GS2<:Grid,G1<:Grid,G2<:Grid,V<:AbstractVector,
                    I<:Integer,DV<:AbstractMatrix,FV<:AbstractVector,M<:AbstractMatrix,FA<:Factorization,FTV<:AbstractVector}
    ε²::FT
    μ::FT
    ϱ::FT
    Δt::FT
    H::F2
    Hx::F1
    Hy::F1
    f::FT
    β::FT
    τx::F2
    τy::F2
    τx_x::F1
    τx_y::F1
    τy_x::F1
    τy_y::F1
    g_sfc1::GS1
    g_sfc2::GS2
    g1::G1
    g2::G2
    σ::V
    nσ::I
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
# Constructors for ModelSetup3D
################################################################################

function ModelSetup3D(ε², μ, ϱ, Δt, f, β, H::Function, τx::Function, τy::Function, g_sfc1, nσ=0, chebyshev=false)
    # second order surface mesh
    g_sfc2 = Grid(2, g_sfc1)

    # 3D mesh
    g1, g2, σ = generate_wedge_cols(g_sfc1, g_sfc2, nσ=nσ, chebyshev=chebyshev)
    nσ = length(σ)

    # convert functions to FE fields
    H = FEField(H, g_sfc2)
    τx = FEField(τx, g_sfc2)
    τy = FEField(τy, g_sfc2)
    # H = FEField(H, g_sfc1)
    # τx = FEField(τx, g_sfc1)
    # τy = FEField(τy, g_sfc1)
    H1 = FEField(H[1:g_sfc1.np], g_sfc1)
    τx1 = FEField(τx[1:g_sfc1.np], g_sfc1)
    τy1 = FEField(τy[1:g_sfc1.np], g_sfc1)

    # store their gradients as DG fields
    Hx = DGField([∂x(H, g_sfc1.p[g_sfc1.t[k, i], :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    Hy = DGField([∂y(H, g_sfc1.p[g_sfc1.t[k, i], :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τx_x = DGField([∂x(τx, g_sfc1.p[g_sfc1.t[k, i], :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τx_y = DGField([∂y(τx, g_sfc1.p[g_sfc1.t[k, i], :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_x = DGField([∂x(τy, g_sfc1.p[g_sfc1.t[k, i], :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_y = DGField([∂y(τy, g_sfc1.p[g_sfc1.t[k, i], :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    # plots
    quick_plot(H, L"H", "$out_folder/H.png")
    quick_plot(Hx, L"H_x", "$out_folder/Hx.png")
    quick_plot(Hy, L"H_y", "$out_folder/Hy.png")
    f_over_H = FEField(x->f + β*x[2], H.g)/(H + FEField(1e-5, H.g))
    quick_plot(f_over_H, L"f/H", "$out_folder/f_over_H.png", vmax=6)
    curl = (τy_x - τx_y)*H - (τy*Hx - τx*Hy)
    quick_plot(curl, L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", "$out_folder/curl.png")

    # derivative matrices
    Dxs, Dys = get_b_gradient_matrices(g1, g2, σ, H, Hx, Hy) 
    # Dxs = zeros(2, 2)
    # Dys = zeros(2, 2)
    
    # baroclinc LHS for each node column on first order grid
    baroclinic_LHSs = [i ∉ g_sfc1.e["bdy"] ? get_baroclinic_LHS(σ*H[i], ε², f + β*g_sfc1.p[i, 2]) : lu(sparse([1;;])) for i=1:g_sfc1.np]

    # get transport ω and χ
    ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = get_transport_ω_and_χ(baroclinic_LHSs, g_sfc1, σ, H, ε², showplots=true)
    ωx_Ux_bot = FEField([ωx_Ux[i, 1] for i=1:g_sfc1.np], g_sfc1)
    ωy_Ux_bot = FEField([ωy_Ux[i, 1] for i=1:g_sfc1.np], g_sfc1)

    # bottom drag coefficients
    r_sym  = ωy_Ux_bot/H1^3
    r_asym = ωx_Ux_bot/H1^3
    # r_sym = FEField(0.1./H1.values/ε², g_sfc1)
    # r_asym = FEField(0.0, g_sfc1)

    # barotropic LHS
    barotropic_LHS = get_barotropic_LHS(r_sym, r_asym, f, β, H, Hx, Hy, ε²)

    # get ω_τ's
    ωx_τx, ωy_τx, χx_τx, χy_τx = get_wind_ω_and_χ(baroclinic_LHSs, g_sfc1, σ, H, ε², showplots=true)
    ωx_τx_bot = FEField([ωx_τx[i, 1] for i=1:g_sfc1.np], g_sfc1)
    ωy_τx_bot = FEField([ωy_τx[i, 1] for i=1:g_sfc1.np], g_sfc1)
    ωx_τy_bot = -ωy_τx_bot
    ωy_τy_bot = ωx_τx_bot
    ωx_τ_bot = (τx1*ωx_τx_bot + τy1*ωx_τy_bot)/H1
    ωy_τ_bot = (τx1*ωy_τx_bot + τy1*ωy_τy_bot)/H1
    quick_plot(ωx_τ_bot*H1, L"H \omega^x_\tau(-H)", "$out_folder/omegax_tau_bot.png")
    quick_plot(ωy_τ_bot*H1, L"H \omega^y_\tau(-H)", "$out_folder/omegay_tau_bot.png")

    # barotropic RHS due to wind stress
    barotropic_RHS_τ = get_barotropic_RHS_τ(H, Hx, Hy, τx, τy, τx_y, τy_x, ωx_τ_bot, ωy_τ_bot, ε²)

    return ModelSetup3D(ε², μ, ϱ, Δt, H, Hx, Hy, f, β, τx, τy, τx_x, τx_y, τy_x, τy_y, g_sfc1, g_sfc2, g1, g2, σ, nσ, Dxs, Dys, 
                        baroclinic_LHSs, ωx_Ux, ωy_Ux, χx_Ux, χy_Ux, barotropic_LHS, ωx_τx, ωy_τx, χx_τx, χy_τx, barotropic_RHS_τ)
end