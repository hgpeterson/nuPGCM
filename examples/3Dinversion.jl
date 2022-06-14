using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_basin_geometry()
    # geometry type
    # geo = "square"
    geo = "circle"

    # bathymetry type
    # bath = "flat"
    bath = "tub"

    # resolution
    # res = 1
    res = 2
    # res = 3

    # load horizontal mesh
    p, t, e = load_mesh("../meshes/$(geo)$res.h5")
    np = size(p, 1)

    # widths of basin
    Lx = 5e6
    Ly = 5e6

    # rescale p
    p[:, 1] *= Lx
    p[:, 2] *= Ly
    ξ = p[:, 1]
    η = p[:, 2]

    # depth H
    H₀ = 4e3
    Δ = Lx/5 # width of gaussian for bathtub
    G(x) = 1 - exp(-x^2/(2*Δ^2)) # gaussian for bathtub
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    if bath == "flat"
        # flat bottom
        H = H₀*ones(np)
        Hx = zeros(np)
        Hy = zeros(np)
    elseif bath == "tub"
        if geo == "square"
            # square bathtub
            H = @. H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) + 100
            Hx = @. H₀*Gx(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gx(Lx - ξ)*G(Ly + η)*G(Ly - η)
            Hy = @. H₀*G(Lx + ξ)*G(Lx - ξ)*Gx(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gx(Ly - η)
        elseif geo == "circle"
            # circular bathtub (radius = Lx)
            H = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + 40
            Hx = @. H₀*Gx(sqrt(ξ^2 + η^2) - Lx)*ξ/sqrt(ξ^2 + η^2)
            Hy = @. H₀*Gx(sqrt(ξ^2 + η^2) - Lx)*η/sqrt(ξ^2 + η^2)
        end
    end

    return p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy
end

function setup_model()
    # use bl theory?
    bl = false

    # ref density
    ρ₀ = 1000.

    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # vertical coordinate
    nσ = 2^8
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  

    # coriolis parameter f = f₀ + βη
    Ω = 2π/86400
    a = 6.378e6
    # ϕ = 45*π/180
    ϕ = 37*π/180
    # ϕ = 0
    f₀ = 2Ω*sin(ϕ)
    β = 2Ω*cos(ϕ)/a

    # diffusivity and viscosity
    # κ0 = 6e-5
    # κ1 = 2e-3
    # h = 200
    # μ = 1e0
    # κ = zeros(np, nσ)
    # for i=1:nσ
    #     κ[:, i] = @. κ0 + κ1*exp(-H*(σ[i] + 1)/h)
    # end
    # ν = μ*κ
    ν = 1e-1*ones(np, nσ)
    κ = 1e-1*ones(np, nσ)

    # stratification
    N² = 1e-6*ones(np, nσ)

    # model setup struct
    m = ModelSetup3DPG(bl, ρ₀, f₀, β, Lx, Ly, p, t, e, σ, H, Hx, Hy, ν, κ, N², 0.)

    # plot H
    plot_horizontal(p, t, H; clabel=L"Depth $H$ (m)")
    savefig("images/H.png")
    println("images/H.png")
    plt.close()

    # plot Hx
    plot_horizontal(p, t, Hx; clabel=L"Slope $\partial_x H$ (-)")
    savefig("images/Hx.png")
    println("images/Hx.png")
    plt.close()

    # plot Hy
    plot_horizontal(p, t, Hy; clabel=L"Slope $\partial_y H$ (-)")
    savefig("images/Hy.png")
    println("images/Hy.png")
    plt.close()

    # plot f/H
    f_over_H = @. (f₀ + β*η)/(H + eps())
    plot_horizontal(p, t, f_over_H; vext=1e-7, clabel=L"$f/H$ (s$^{-1}$ m$^{-1}$)")
    savefig("images/f_over_H.png")
    println("images/f_over_H.png")
    plt.close()

    # plot baroclinic components 
    plot_horizontal(p, t, m.τξ_tξ[:, 1]; clabel=L"Symmetric bottom stress $\tau^\xi_{t\xi}$ (kg m$^{-3}$ s$^{-1}$)", contours=false)
    savefig("images/tau_xi_t.png")
    println("images/tau_xi_t.png")
    plt.close()
    plot_horizontal(p, t, m.τη_tξ[:, 1]; clabel=L"Anti-symmetric bottom stress $\tau^\eta_{t\xi}$ (kg m$^{-3}$ s$^{-1}$)", contours=false)
    savefig("images/tau_eta_t.png")
    println("images/tau_eta_t.png")
    plt.close()
    plot_horizontal(p, t, m.τξ_wξ[:, 1]; clabel=L"Symmetric bottom stress $\tau^\xi_{w\xi}$ (-)", contours=false)
    savefig("images/tau_xi_w.png")
    println("images/tau_xi_w.png")
    plt.close()
    plot_horizontal(p, t, m.τη_wξ[:, 1]; clabel=L"Anti-symmetric bottom stress $\tau^\eta_{w\xi}$ (-)", contours=false)
    savefig("images/tau_eta_w.png")
    println("images/tau_eta_w.png")
    plt.close()

    return m
end

function invert3D(m)
    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # buoyancy field
    b = zeros(np, m.nσ)
    for j=1:m.nσ
        b[:, j] .= m.N²[:, j].*m.H*m.σ[j] + 0.1*m.N²[:, j].*m.H*exp(-(m.σ[j] + 1)/0.1)
    end

    # # plot b slice
    # s = ModelState3DPG(b, zeros(1), zeros(1, 1), zeros(1, 1), zeros(1, 1), [1])
    # ξ_slice = (-Lx + 1e3):Lx/2^8:(Lx - 1e3)
    # η₀ = 0
    # ax = plot_ξ_slice(m, s, b, ξ_slice, η₀; clabel=L"Buoyancy $b$ (m s$^{-2}$)", contours=false)
    # ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    # ax.set_ylim([-4, 0])
    # savefig("images/b_slice.png")
    # println("images/b_slice.png")
    # plt.close()
    # error()

    # integrals of buoyancy gradients on rhs
    bσ_x = zeros(np, m.nσ)
    bσ_y = zeros(np, m.nσ)
    for i=1:np
        bσ_x[i, :] = -m.σ*Hx[i].*differentiate(b[i, :], m.σ)/H[i] 
        bσ_y[i, :] = -m.σ*Hy[i].*differentiate(b[i, :], m.σ)/H[i]
    end
    rhs_x = m.Cξ*b + m.M*bσ_x
    rhs_y = m.Cη*b + m.M*bσ_y
    for i=1:np
        rhs_x[i, :] .*= m.ρ₀*m.ν[i, :]/(m.f₀ + m.β*η[i])
        rhs_y[i, :] .*= m.ρ₀*m.ν[i, :]/(m.f₀ + m.β*η[i])
    end

    # stress due to buoyancy gradients
    baroclinic_RHSs_b = zeros(np, 2*m.nσ)
    for i=1:np
        # if i in e
        #     continue
        # else
        #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
        # end
        baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
    end
    vξ_b, vη_b = get_vξ_vη(m.baroclinic_LHSs, baroclinic_RHSs_b)
    τξ_b = m.M_LU\vξ_b
    τη_b = m.M_LU\vη_b

    # bottom stress due buoyancy gradients
    τξ_b_bot = τξ_b[:, 1]
    τη_b_bot = τη_b[:, 1]
    plot_horizontal(p, t, τξ_b_bot; clabel=L"Buoyancy bottom stress $\tau^\xi_b$ (kg m$^{-1}$ s$^{-2}$)")
    savefig("images/tau_xi_b.png")
    println("images/tau_xi_b.png")
    plt.close()
    plot_horizontal(p, t, τη_b_bot; clabel=L"Buoyancy bottom stress $\tau^\eta_b$ (kg m$^{-1}$ s$^{-2}$)")
    savefig("images/tau_eta_b.png")
    println("images/tau_eta_b.png")
    plt.close()

    # buoyancy integral for JEBAR term
    γ = zeros(np)
    for i=1:np
        γ[i] = -H[i]^2*trapz(m.σ.*b[i, :], m.σ)
    end
    plot_horizontal(p, t, γ; clabel=L"Buoyancy integral $\gamma$ (m$^{3}$ s$^{-2}$)")
    savefig("images/gamma.png")
    println("images/gamma.png")
    plt.close()

    # wind stress
    # τξ₀ = @. -0.1*cos(π*η/Ly)
    τξ₀ = zeros(np)
    τη₀ = zeros(np)

    # bottom stress due to wind stress
    τξ_w_bot = m.τξ_wξ[:, 1]
    τη_w_bot = m.τη_wξ[:, 1]

    # full τ
    τξ = @. τξ₀ - (τξ₀*τξ_w_bot + τη₀*τξ_w_bot) - τξ_b_bot
    τη = @. τη₀ - (τξ₀*τη_w_bot - τη₀*τη_w_bot) - τη_b_bot

    # get barotropic_RHS
    barotropic_RHS = get_barotropic_RHS(m, γ, τξ, τη)

    # solve
    Ψ = m.barotropic_LHS\barotropic_RHS

    # # Uξ and Uη
    # Uξ = ∂ξ(m, Ψ)
    # Uη = ∂η(m, Ψ)

    # # get τ
    # τ = zeros(2, np, nσ)
    # for j=1:nσ
    #     τ[1, :, j] = τ_b[1, :, j] + τξ₀.(ξ, η)*m.τ_wξ[1, :, j] + τη₀.(ξ, η)*m.τ_wξ[2, :, j] + Uξ*m.τ_tξ[1, :, j] + Uη*m.τ_tξ[2, :, j]
    #     τ[2, :, j] = τ_b[2, :, j] + τξ₀.(ξ, η)*m.τ_wξ[1, :, j] - τη₀.(ξ, η)*m.τ_wξ[2, :, j] + Uξ*m.τ_tξ[1, :, j] - Uη*m.τ_tξ[2, :, j]
    # end

    # # convert to uξ, uη
    # u = get_u(m, τ)

    # # compute uσ
    # div = ∂ξ(m, u[1, :, :]) + ∂η(m, u[2, :, :])
    # uσ = zeros(np, nσ)
    # for i=1:np
    #     uσ[i, :] = cumtrapz(-div[i, :], m.σ)
    # end

    # plot Ψ
    fig, ax, im = plot_horizontal(p, t, Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)")
    savefig("images/psi.png")
    println("images/psi.png")
    plt.close()

    # # plot wind stress
    # y = -Ly:2*Ly/100:Ly
    # fig, ax = subplots(figsize=(1.955, 3.167))
    # ax.axvline(0, c="k", lw=0.5, ls="-")
    # ax.plot(τξ₀.(0, y), y/1e3)
    # ax.set_xlabel(L"Wind stress $\tau^\xi_0$ (N m$^{-2}$)")
    # ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
    # ax.spines["left"].set_visible(false)
    # ax.set_xlim([-0.15, 0.15])
    # ax.set_xticks(-0.15:0.05:0.15)
    # savefig("images/tau.png")
    # println("images/tau.png")
    # plt.close()

    s = ModelState3DPG(b, Ψ, zeros(2, 2), zeros(2, 2), zeros(2, 2), [1])
    return s
end

# function plot_curl_τ_H()
#     # basin geo
#     p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

#     # linear basis
#     C₀ = get_linear_basis_coeffs(p, t)

#     # wind stress
#     τξ = @. -0.1*cos(π*η/Ly)

#     # functions 
#     H_func(ξ, η, k)  = evaluate(H,  [ξ, η], p, t, C₀, k)
#     τξ_func(ξ, η, k) = evaluate(τξ, [ξ, η], p, t, C₀, k)

#     # curl
#     curl_τ(ξ, η, k) = -∂η(τξ, [ξ, η], k, p, t, C₀)/H_func(ξ, η, k) + τξ_func(ξ, η, k)/H_func(ξ, η, k)^2*∂η(H, [ξ, η], k, p, t, C₀)
#     # curl_τ(ξ, η, k) = -∂η(τξ, [ξ, η], k, p, t, C₀)
#     # curl_τ(ξ, η, k) = ∂ξ(H, [ξ, η], k, p, t, C₀)

#     # evaluate at triangle centers
#     curl = zeros(size(t, 1))
#     for k=1:size(t, 1)
#         # triangle center
#         p₀ = sum(p[t[k, :], :], dims=1)/3

#         # curl
#         c = curl_τ(p₀[1], p₀[2], k)
#         if isnan(c)
#             curl[k] = Inf
#         else
#             curl[k] = c
#         end
#     end

#     # plot
#     fig, ax, im = tplot(p/1e3, t, ; vext=30)
#     fig, ax = subplots()
#     im = ax.tripcolor(p[:, 1]/1e3, p[:, 2]/1e3, t .- 1, log.(abs.(curl)), vmin=-30, vmax=-10, shading="flat")
#     cb = colorbar(im, ax=ax, label=L"\log | \nabla \times (\tau_0 / H) |", extend="both")
#     ax.set_xlabel(L"Horizontal coordinate $\xi$ (km)")
#     ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
#     ax.set_yticks(-5000:2500:5000)
#     ax.spines["left"].set_visible(false)
#     ax.spines["bottom"].set_visible(false)
#     ax.axis("equal")
#     savefig("images/curl_tau_H.png")
#     println("images/curl_tau_H.png")
#     plt.close()

#     return curl
# end

m = setup_model()
s = invert3D(m)
# curl = plot_curl_τ_H()

# fig, ax, im = plot_horizontal(m.p, m.t, s.Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)", vext=6)
# ax.set_yticklabels(0:2500:10000)
# # fig, ax, im = plot_horizontal(m.p, m.t, s.Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)", vext=10, ncontours=5)
# savefig("images/psi.pdf")
# println("images/psi.pdf")
# savefig("images/psi.png")
# println("images/psi.png")
# plt.close()