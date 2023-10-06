################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState3D{I<:Integer,F1<:AbstractField,F2<:AbstractField,FS1<:AbstractField}
    # buoyancy
	b::F2

    # vorticity
	ωx::F1
	ωy::F1

    # streamfunction
    χx::F1
    χy::F1

    # barotropic streamfunction
    Ψ::FS1

    # iteration
    i::I
end

struct ModelSetup3D{FT<:AbstractFloat,F1<:AbstractField,F2<:AbstractField,F3<:AbstractField,GS<:Grid,G<:Grid,GC<:Grid,
                    V<:AbstractVector,IN<:AbstractVector,I<:Integer,DV<:AbstractMatrix,FV<:AbstractVector,M<:AbstractMatrix,FA<:Factorization,
                    FTV<:AbstractVector,HM<:SparseMatrixCSC,A<:AbstractArray}
    ε²::FT
    μ::FT
    ϱ::FT
    Δt::FT
    H::F1
    Hx::F2
    Hy::F2
    f::FT
    β::FT
    τx::F1
    τy::F1
    τx_x::F2
    τx_y::F2
    τy_x::F2
    τy_y::F2
    ν::F3
    ν_bot::F1
    κ::F3
    g_sfc1::GS
    g_sfc2::GS
    g1::G
    g2::G
    g_col::GC
    in_nodes1::IN
    in_nodes2::IN
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
    HM::HM
    Ax::A
    Ay::A
    advection::Bool
    K_stab::HM
end

################################################################################
# Constructors for ModelSetup3D
################################################################################

function ModelSetup3D(ε², μ, ϱ, Δt, f, β, H_func::Function, τx_func::Function, τy_func::Function, 
                      ν_func::Function, κ_func::Function, g_sfc1; nσ=0, chebyshev=false, advection=true)
    # second order surface mesh
    g_sfc2 = add_midpoints(g_sfc1)

    # 3D mesh
    g1, g2, σ = generate_wedge_cols(g_sfc1, g_sfc2, nσ=nσ, chebyshev=chebyshev)

    # 1D grid
    nσ = length(σ)
    p = σ
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g_col = Grid(Line(order=1), p, t, e)

    # convert functions to FE fields
    H = FEField(H_func, g_sfc2)
    τx = FEField(τx_func, g_sfc2)
    τy = FEField(τy_func, g_sfc2)
    ν = FEField([ν_func(g2.p[i, 3], H[get_i_sfc(i, nσ)]) for i=1:g2.np], g2)
    κ = FEField([κ_func(g2.p[i, 3], H[get_i_sfc(i, nσ)]) for i=1:g2.np], g2)

    # pass to next constuctor
    return ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, ν, κ, g_sfc1, g_sfc2, g1, g2, g_col, advection=advection)
end

function ModelSetup3D(ε², μ, ϱ, Δt, f, β, H::AbstractField, τx::AbstractField, τy::AbstractField, ν::AbstractField, κ::AbstractField, 
                      g_sfc1, g_sfc2, g1, g2, g_col; advection=true)

    # unpack
    σ = g_col.p
    nσ = g_col.np
    @printf("BL thickness: %1.1e\n", √(2*ε²))
    @printf("Bottom res:   %1.1e\n", σ[2] - σ[1])

    # fields for later
    τx1 = FEField(τx[1:g_sfc1.np], g_sfc1)
    τy1 = FEField(τy[1:g_sfc1.np], g_sfc1)
    ν_bot = FEField(ν[get_i_bot.(1:g_sfc1.np, nσ)], g_sfc1)

    # store gradients as DG fields
    Hx = DGField([∂x(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    Hy = DGField([∂y(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τx_x = DGField([∂x(τx, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τx_y = DGField([∂y(τx, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_x = DGField([∂x(τy, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_y = DGField([∂y(τy, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    # plots
    quick_plot(H, L"H", "$out_folder/H.png")
    quick_plot(Hx, L"H_x", "$out_folder/Hx.png")
    quick_plot(Hy, L"H_y", "$out_folder/Hy.png")
    f_over_H = FEField(x->f + β*x[2], H.g)/(H + FEField(1e-5, H.g))
    quick_plot(f_over_H, L"f/H", "$out_folder/f_over_H.png", vmax=6)
    curl = (τy_x - τx_y)*H - (τy*Hx - τx*Hy)
    quick_plot(curl, L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", "$out_folder/curl.png")

    # indices of nodes in interior
    in_nodes1 = findall(i -> i ∉ g_sfc1.e["bdy"], 1:g_sfc1.np)
    in_nodes2 = findall(i -> i ∉ g_sfc2.e["bdy"], 1:g_sfc2.np)

    # derivative matrices
    Dxs, Dys = build_b_gradient_matrices(g1, g2, σ, H, Hx, Hy) 
    # Dxs = [1;;]
    # Dys = [1;;]
    
    # baroclinc LHS for each node column on first order grid
    baroclinic_LHSs = [build_baroclinic_LHS(g_col, ν[get_col_inds(i, nσ)], H[i], ε², f + β*g_sfc1.p[i, 2]) for i ∈ in_nodes1]

    # get transport ω and χ
    ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = solve_baroclinic_transport(baroclinic_LHSs, g_sfc1, g_col, in_nodes1, H, showplots=true)
    νωx_Ux_bot = ν_bot*FEField(ωx_Ux[:, 1], g_sfc1)
    νωy_Ux_bot = ν_bot*FEField(ωy_Ux[:, 1], g_sfc1)

    # barotropic LHS
    barotropic_LHS = build_barotropic_LHS(νωx_Ux_bot, νωy_Ux_bot, f, β, H, Hx, Hy, ε²)
    # barotropic_LHS = lu([1;;])

    # get ω_τ's
    ωx_τx, ωy_τx, χx_τx, χy_τx = solve_baroclinic_wind(baroclinic_LHSs, g_sfc1, g_col, in_nodes1, ε², showplots=true)
    νωx_τx_bot = ν_bot*FEField(ωx_τx[:, 1], g_sfc1)
    νωy_τx_bot = ν_bot*FEField(ωy_τx[:, 1], g_sfc1)
    νωx_τ_bot = τx1*νωx_τx_bot - τy1*νωy_τx_bot
    νωy_τ_bot = τx1*νωy_τx_bot + τy1*νωx_τx_bot
    quick_plot(νωx_τ_bot, L"\nu\omega^x_\tau|_{-H}", "$out_folder/nu_omegax_tau_bot.png")
    quick_plot(νωy_τ_bot, L"\nu\omega^y_\tau|_{-H}", "$out_folder/nu_omegay_tau_bot.png")

    # barotropic RHS due to wind stress
    barotropic_RHS_τ = build_barotropic_RHS_τ(H, Hx, Hy, τx, τy, τx_y, τy_x, νωx_τ_bot, νωy_τ_bot, ε²)
    # barotropic_RHS_τ = []

    # HM and advection arrays for evolution
    if advection
        HM = build_HM(g2, H, nσ)
        Ax, Ay = build_advection_arrays(g1, g2)
        # K_stab = build_K_stab(g2, H, Hx, Hy, nσ)
        K_stab = spzeros(g2.np, g2.np)
    else
        HM = spzeros(g2.np, g2.np)
        Ax = Ay = zeros(1, 1, 1, 1)
        K_stab = spzeros(g2.np, g2.np)
    end

    return ModelSetup3D(ε², μ, ϱ, Δt, H, Hx, Hy, f, β, τx, τy, τx_x, τx_y, τy_x, τy_y, ν, ν_bot, κ, g_sfc1, g_sfc2, g1, g2, g_col,
                        in_nodes1, in_nodes2, σ, nσ, Dxs, Dys, baroclinic_LHSs, ωx_Ux, ωy_Ux, χx_Ux, χy_Ux, barotropic_LHS, 
                        ωx_τx, ωy_τx, χx_τx, χy_τx, barotropic_RHS_τ, HM, Ax, Ay, advection, K_stab)
end