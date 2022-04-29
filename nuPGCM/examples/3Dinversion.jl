using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_basin_geometry()
    # load horizontal mesh
    # p, t, e = load_mesh("../meshes/square1.h5")
    # p, t, e = load_mesh("../meshes/square2.h5")
    # p, t, e = load_mesh("../meshes/square3.h5")
    # p, t, e = load_mesh("../meshes/circle1.h5")
    # p, t, e = load_mesh("../meshes/circle2.h5")
    p, t, e = load_mesh("../meshes/circle3.h5")
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
    # Δ = Lx/5
    # G(x) = 1 - exp(-x^2/(2*Δ^2))
    # Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    # H_func(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η)
    # Hx_func(ξ, η) = H₀*Gx(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gx(Lx - ξ)*G(Ly + η)*G(Ly - η)
    # Hy_func(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*Gx(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gx(Ly - η)

    H₀ = 4e3
    R = Lx
    Δ = R/5
    G(x) = 1 - exp(-x^2/(2*Δ^2))
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    H_func(ξ, η) = H₀*G(sqrt(ξ^2 + η^2) - R)
    Hx_func(ξ, η) = H₀*Gx(sqrt(ξ^2 + η^2) - R)*ξ/sqrt(ξ^2 + η^2)
    Hy_func(ξ, η) = H₀*Gx(sqrt(ξ^2 + η^2) - R)*η/sqrt(ξ^2 + η^2)

    return p, t, e, np, Lx, Ly, ξ, η, H_func, Hx_func, Hy_func 
end

function setup_model()
    # use bl theory?
    bl = false

    # ref density
    ρ₀ = 1000.

    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H_func, Hx_func, Hy_func = get_basin_geometry()

    # linear basis
    C₀ = get_linear_basis_coeffs(p, t)

    # vertical coordinate
    nσ = 2^8
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  

    # coriolis parameter f = f₀ + βη
    f₀ = 0
    β = 1e-11
    f_func(ξ, η) = f₀ + β*η
    fy_func(ξ, η) = β

    # diffusivity and viscosity
    # κ0 = 6e-5
    # κ1 = 2e-3
    # h = 200
    # μ = 1e0
    # κ = zeros(np, nσ)
    # for i=1:nσ
    #     κ[:, i] = @. κ0 + κ1*exp(-H_func.(ξ, η)*(σ[i] + 1)/h)
    # end
    # ν = μ*κ
    ν = 1e-3*ones(np, nσ)
    κ = 1e-3*ones(np, nσ)

    # stratification
    N² = 1e-6*ones(np, nσ)

    # model setup struct
    m = ModelSetup3DPG(bl, ρ₀, f_func, fy_func, Lx, Ly, p, t, e, σ, H_func, Hx_func, Hy_func, ν, κ, N², 0.)

    # plot H
    plot_horizontal(p, t, H_func.(ξ, η); clabel=L"$H$ (m)")
    savefig("H.png")
    println("H.png")
    plt.close()

    # plot Hx
    plot_horizontal(p, t, Hx_func.(ξ, η); clabel=L"$\partial_x H$ (-)")
    savefig("Hx.png")
    println("Hx.png")
    plt.close()

    # plot Hy
    plot_horizontal(p, t, Hy_func.(ξ, η); clabel=L"$\partial_y H$ (-)")
    savefig("Hy.png")
    println("Hy.png")
    plt.close()

    # plot f/H
    f_over_H = @. f_func(ξ, η)/(H_func(ξ, η) + eps())
    plot_horizontal(p, t, f_over_H; vext=1e-8, clabel=L"$f/H$ (s m$^{-1}$)")
    savefig("f_over_H.png")
    println("f_over_H.png")
    plt.close()

    # plot baroclinic components 
    plot_horizontal(p, t, m.τ_tξ[1, :, 1]; clabel=L"Symmetric bottom stress $\tau^\xi_{t\xi}$ (s$^{-1}$)")
    savefig("tau_xi_t.png")
    println("tau_xi_t.png")
    plt.close()
    plot_horizontal(p, t, m.τ_tξ[2, :, 1]; clabel=L"Anti-symmetric bottom stress $\tau^\eta_{t\xi}$ (s$^{-1}$)")
    savefig("tau_eta_t.png")
    println("tau_eta_t.png")
    plt.close()
    plot_horizontal(p, t, m.τ_wξ[1, :, 1]; clabel=L"Symmetric bottom stress $\tau^\xi_{w\xi}$ (s$^{-1}$)")
    savefig("tau_xi_w.png")
    println("tau_xi_w.png")
    plt.close()
    plot_horizontal(p, t, m.τ_wξ[2, :, 1]; clabel=L"Anti-symmetric bottom stress $\tau^\eta_{w\xi}$ (s$^{-1}$)")
    savefig("tau_eta_w.png")
    println("tau_eta_w.png")
    plt.close()

    return m
end

function invert3D(m)
    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H_func, Hx_func, Hy_func = get_basin_geometry()

    # # buoyancy field
    # b = zeros(np, nσ)

    # # buoyancy gradients
    # ∂b∂x = zeros(np, nσ)
    # ∂b∂y = zeros(np, nσ)
    # for i=1:nσ
    #     println("i = $i / $nσ")
    #     for j=1:np
    #         ∂b∂x[:, i] .+= ∂ξ(b, p[j, :], p, t, C₀)
    #         ∂b∂y[:, i] .+= ∂η(b, p[j, :], p, t, C₀)
    #     end
    # end
    # for i=1:np
    #     println("i = $i / $np")
    #     ∂b∂x[i, :] .-= σ*Hx[i].*differentiate(b[i, :], σ)/H[i]
    #     ∂b∂y[i, :] .-= σ*Hy[i].*differentiate(b[i, :], σ)/H[i]
    # end

    # JEBAR term
    JEBAR(ξ, η) = 0

    # wind stress
    τ₀ = 0.1 # kg m⁻¹ s⁻² 
    τξ₀(ξ, η) = -τ₀*cos(π*η/Ly)
    τη₀(ξ, η) = 0

    # curl of wind stress [∂ξ(τη/H) - ∂η(τξ/H)]
    curl_τ₀(ξ, η) = -τ₀*π/Ly*sin(π*η/Ly)/H_func(ξ, η) - τξ₀(ξ, η)*Hy_func(ξ, η)/H_func(ξ, η)^2  

    # curl of bottom stress due to wind stress
    curl_τ_w_bot(ξ, η) = 0

    # # stress due to buoyancy gradients
    # baroclinic_RHSs_b = zeros(np, 2*nσ)
    # @inbounds for i=1:np
    #     if i in e
    #         continue
    #     else
    #         rhs_x = @. m.ν[i, :]/m.ρ₀/m.f[i]*∂b∂x[i, :]
    #         rhs_y = @. m.ν[i, :]/m.ρ₀/m.f[i]*∂b∂y[i, :]
    #         baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x, rhs_y, 0, 0, 0, 0)
    #     end
    # end
    # τ_b = get_τ(baroclinic_LHSs, baroclinic_RHSs_b)

    # curl of bottom stress due buoyancy gradients
    curl_τ_b_bot(ξ, η) = 0

    # right-hand-side forcing
    F(ξ, η) = JEBAR(ξ, η) + 1/m.ρ₀*(curl_τ₀(ξ, η) - curl_τ_w_bot(ξ, η) - curl_τ_b_bot(ξ, η))

    # get barotropic_RHS
    barotropic_RHS = get_barotropic_RHS(m, F)

    # solve
    Ψ = m.barotropic_LHS\barotropic_RHS

    # plot Ψ
    plot_horizontal(p, t, Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)")
    savefig("psi.png")
    println("psi.png")
    plt.close()

    # plot wind stress
    y = -Ly:2*Ly/100:Ly
    fig, ax = subplots(figsize=(1.955, 3.167))
    ax.axvline(0, c="k", lw=0.5, ls="-")
    ax.plot(τξ₀.(0, y), y/1e3)
    ax.set_xlabel(L"Wind stress $\tau^\xi_0$ (N m$^{-2}$)")
    ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
    ax.spines["left"].set_visible(false)
    ax.set_xlim([-0.15, 0.15])
    ax.set_xticks(-0.15:0.05:0.15)
    savefig("tau.png")
    println("tau.png")
    plt.close()
end

m = setup_model()
invert3D(m)