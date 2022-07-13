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
    # res = 1   #  1452 linear nodes,   5677 quadratic nodes
    # res = 2   #  4027 linear nodes,  15899 quadratic nodes
    res = 3   #  9062 linear nodes,  35936 quadratic nodes
    # res = 4   # 36268 linear nodes, 144433 quadratic nodes
    # res = 5   # 74035 linear nodes, 295233 quadratic nodes

    # load horizontal mesh
    p, t, e = load_mesh("../meshes/$(geo)$res.h5")
    # p, t, e = add_midpoints(p, t)
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
    # H₀ = 4e3
    H₀ = 2e3
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
            H = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + 5
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
    nσ = 2^7
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  

    # coriolis parameter f = f₀ + βη
    Ω = 2π/86400
    a = 6.378e6
    # ϕ = 45*π/180
    ϕ = 37*π/180
    # ϕ = 0
    f₀ = 2Ω*sin(ϕ)
    # β = 2Ω*cos(ϕ)/a
    β = 0.

    # diffusivity and viscosity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    μ = 1e0
    κ = zeros(np, nσ)
    for i=1:nσ
        κ[:, i] = @. κ0 + κ1*exp(-H*(σ[i] + 1)/h)
    end
    ν = μ*κ
    # ν = 1e-1*ones(np, nσ)
    # κ = 1e-1*ones(np, nσ)

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
        # b[:, j] .= m.N²[:, j].*m.H*m.σ[j] 
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

    # plot Ψ
    fig, ax, im = plot_horizontal(p, t, Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)")
    savefig("images/psi.png")
    println("images/psi.png")
    plt.close()

    # get τ
    τξ, τη = get_full_τξ_τη(m, τξ_b, τη_b, τξ₀, τη₀, Ψ)

    # convert to uξ, uη
    uξ, uη = get_uξ_uη(m, τξ, τη)
    s = ModelState3DPG(b, Ψ, uξ, uη, zeros(2, 2), [1])

    # # compute uσ
    # div = ∂ξ(m, u[1, :, :]) + ∂η(m, u[2, :, :])
    # uσ = zeros(np, nσ)
    # for i=1:np
    #     uσ[i, :] = cumtrapz(-div[i, :], m.σ)
    # end

    # s = ModelState3DPG(b, Ψ, zeros(2, 2), zeros(2, 2), zeros(2, 2), [1])
    return s
end

function plot_uξ_uη_slice(m, s)
    # plot uξ slice
    ξ_slice = (-m.Lx + 1e3):m.Lx/2^8:(m.Lx - 1e3)
    η₀ = 0
    ax = plot_ξ_slice(m, s, 1e3*s.uξ, ξ_slice, η₀; clabel=L"Zonal velocity $u^x$ ($\times 10^{-3}$ m s$^{-1}$)", contours=false)
    ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    ax.set_ylim([-maximum(m.H)/1e3, 0])
    savefig("images/ux3D.png")
    println("images/ux3D.png")
    plt.close()

    # plot uη slice
    ξ_slice = (-m.Lx + 1e3):m.Lx/2^8:(m.Lx - 1e3)
    η₀ = 0
    ax = plot_ξ_slice(m, s, 1e3*s.uη, ξ_slice, η₀; clabel=L"Meridional velocity $u^y$ ($\times 10^{-3}$ m s$^{-1}$)", contours=false)
    ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    ax.set_ylim([-maximum(m.H)/1e3, 0])
    savefig("images/uy3D.png")
    println("images/uy3D.png")
    plt.close()
end

m3D = setup_model()
s3D = invert3D(m3D)
plot_uξ_uη_slice(m3D, s3D)

println("Done.")