using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SuiteSparse
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_basin_geometry()
    # geometry type
    # geo = "square"
    geo = "circle"

    # bathymetry type
    bath = "flat"
    # bath = "tub"
    # bath = "bump"

    # resolution
    # res = 1   #  1452 linear nodes,   5677 quadratic nodes
    res = 2   #  4027 linear nodes,  15899 quadratic nodes
    # res = 3   #  9062 linear nodes,  35936 quadratic nodes
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
    # H₀ = 2e2
    Δ = Lx/5 # width of gaussian for bathtub
    G(r) = 1 - exp(-r^2/(2*Δ^2)) # gaussian for bathtub
    Gr(r) = r/Δ^2*exp(-r^2/(2*Δ^2))
    G_bump(r) = if r < 4Δ return -exp(-16*Δ^2/(16*Δ^2 - r^2)) else return 0 end 
    Gr_bump(r) = if r < 4Δ return 32*r*Δ^2*G_bump(r)/(16*Δ^2 - r^2)^2 else return 0 end
    if bath == "flat"
        # flat bottom
        H = H₀*ones(np)
        Hx = zeros(np)
        Hy = zeros(np)
    elseif bath == "tub"
        if geo == "square"
            # square bathtub
            H = @. H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) + 100
            Hx = @. H₀*Gr(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gr(Lx - ξ)*G(Ly + η)*G(Ly - η)
            Hy = @. H₀*G(Lx + ξ)*G(Lx - ξ)*Gr(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gr(Ly - η)
        elseif geo == "circle"
            # circular bathtub (radius = Lx)
            H = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + eps()
            Hx = @. H₀*Gr(sqrt(ξ^2 + η^2) - Lx)*ξ/sqrt(ξ^2 + η^2)
            Hy = @. H₀*Gr(sqrt(ξ^2 + η^2) - Lx)*η/sqrt(ξ^2 + η^2)
        end
    elseif bath == "bump"
        if geo == "circle"
            # circular bump
            H = @. H₀*G_bump(sqrt(ξ^2 + η^2) - 0) + 2e3
            Hx = @. H₀*Gr_bump(sqrt(ξ^2 + η^2) - 0)*ξ/sqrt(ξ^2 + η^2)
            Hy = @. H₀*Gr_bump(sqrt(ξ^2 + η^2) - 0)*η/sqrt(ξ^2 + η^2)
        end
    end

    return p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy
end

function setup_model(; plots=true)
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
    # Ω = 2π/86400
    # a = 6.378e6
    # ϕ = 45*π/180
    # # ϕ = 37*π/180
    # # ϕ = 0
    # f₀ = 2Ω*sin(ϕ)
    # β = 2Ω*cos(ϕ)/a
    f₀ = 1e-4
    β = 0.

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
    # ν = 1e-3*ones(np, nσ)
    # κ = 1e-3*ones(np, nσ)
    ν = 1e-1*ones(np, nσ)
    κ = 1e-1*ones(np, nσ)

    # stratification
    N² = 1e-6*ones(np, nσ)

    # model setup struct
    m = ModelSetup3DPG(bl, ρ₀, f₀, β, Lx, Ly, p, t, e, σ, H, Hx, Hy, ν, κ, N², 0.)

    if plots
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
        plot_horizontal(p, t, m.H.^2 .*m.τξ_tξ[:, 1]; clabel=L"Symmetric bottom stress $H^2 \tau^\xi_{t\xi}$ (kg m$^{-1}$ s$^{-1}$)", contours=false)
        savefig("images/tau_xi_t.png")
        println("images/tau_xi_t.png")
        plt.close()
        plot_horizontal(p, t, m.H.^2 .*m.τη_tξ[:, 1]; clabel=L"Anti-symmetric bottom stress $H^2 \tau^\eta_{t\xi}$ (kg m$^{-1}$ s$^{-1}$)", contours=false)
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
    end

    return m
end

function invert3D(m)
    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # buoyancy field
    b = zeros(np, m.nσ)
    # for j=1:m.nσ
    #     b[:, j] .= m.N²[:, j].*m.H*m.σ[j] + 0.1*m.N²[:, j].*m.H*exp(-(m.σ[j] + 1)/0.1)
    #     # b[:, j] .= m.N²[:, j].*m.H*m.σ[j] 
    # end
    for i=1:np
        Δ = 0.9*m.Lx
        if sqrt(ξ[i]^2 + η[i]^2) < Δ
            b[i, :] .= m.N²[i, :]*m.H[i].*m.σ * (1 - 0.1*exp(-Δ^2/(Δ^2 - ξ[i]^2 - η[i]^2)))
        else
            b[i, :] .= m.N²[i, :]*m.H[i].*m.σ
        end
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
    vξ_b, vη_b = get_τξ_τη(m.baroclinic_LHSs, baroclinic_RHSs_b)
    τξ_b = m.M_LU\vξ_b
    τη_b = m.M_LU\vη_b

    # # buoyancy gradients
    # b_x = m.M_LU\(m.Cξ*b)
    # b_y = m.M_LU\(m.Cη*b)
    # for i=1:m.np
    #     b_x[i, :] += -m.σ*m.Hx[i].*differentiate(b[i, :], m.σ)/m.H[i] 
    #     b_y[i, :] += -m.σ*m.Hy[i].*differentiate(b[i, :], m.σ)/m.H[i]
    # end

    # # stress due to buoyancy gradients
    # baroclinic_RHSs_b = zeros(m.np, 2*m.nσ)
    # for i=1:np
    #     coeff = m.ρ₀*m.ν[i, :]./(m.f₀ .+ m.β*m.p[i, 2])
    #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(coeff.*b_x[i, :], coeff.*b_y[i, :], 0, 0, 0, 0)
    # end
    # τξ_b, τη_b = get_τξ_τη(m.baroclinic_LHSs, baroclinic_RHSs_b)

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
    ξ_slice = (-m.Lx + 1e4):m.Lx/2^7:(m.Lx - 1e4)
    η₀ = 0

    # plot uξ slice
    ax = plot_ξ_slice(m, s, 1e3*s.uξ, ξ_slice, η₀; clabel=L"Zonal velocity $u^\xi$ ($\times 10^{-3}$ m s$^{-1}$)", contours=false)
    ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    ax.set_ylim([-maximum(m.H)/1e3, 0])
    savefig("images/uxi3D.png")
    println("images/uxi3D.png")
    plt.close()

    # plot uη slice
    ax = plot_ξ_slice(m, s, 1e3*s.uη, ξ_slice, η₀; clabel=L"Meridional velocity $u^\eta$ ($\times 10^{-3}$ m s$^{-1}$)", contours=false)
    ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    ax.set_ylim([-maximum(m.H)/1e3, 0])
    savefig("images/ueta3D.png")
    println("images/ueta3D.png")
    plt.close()
end

function plot_Ψ_error()
    # compute effective 2D Ψ
    m2D = load_setup_2D("../output/setup2D.h5")
    s2D = load_state_2D("../output/state2D.h5")
    Ψ2D = zeros(m2D.nξ)
    Uη = zeros(m2D.nξ)
    for i=1:m2D.nξ
        Uη[i] = m2D.H[i]*trapz(s2D.uη[i, :], m2D.σ)
    end
    Ψ2D = cumtrapz(Uη, m2D.ξ) .- trapz(Uη, m2D.ξ)

    # compute 3D Ψ along (ξ, 0) slice
    Ψ3D = zeros(m2D.nξ)
    for i=1:m2D.nξ-1
        Ψ3D[i] = fem_evaluate(m3D, s3D.Ψ, m2D.ξ[i], 0)
    end

    # error
    abs_err = abs.(Ψ3D - Ψ2D)/1e6
    println(@sprintf("%d km", m3D.Lx/sqrt(m3D.np)/1e3))
    println(@sprintf("Max Abs. Err.: %1.16e Sv (%d km)", maximum(abs_err), m2D.ξ[argmax(abs_err)]/1e3))
    println(@sprintf("Max Ψ:         %1.1e Sv (%d km)", maximum(Ψ2D)/1e6, m2D.ξ[argmax(Ψ2D)]/1e3))

    fig, ax = subplots(2, 1, figsize=(19/6, 2*19/6/1.62), sharex=true)
    ax[1].set_ylabel(L"Streamfunction $\Psi$ (Sv)")
    ax[2].set_ylabel("Absolute Error (Sv)")
    ax[2].set_xlabel(L"Zonal distance $x$ (km)")
    ax[1].set_title(latexstring(L"Res $\approx$ ", @sprintf("%d km", m3D.Lx/sqrt(m3D.np)/1e3)))
    ax[1].plot(m2D.ξ/1e3, Ψ2D/1e6, "tab:orange", label="2D")
    ax[1].plot(m2D.ξ/1e3, Ψ3D/1e6, "k--", lw=0.5, label="3D")
    ax[1].legend()
    ax[2].semilogy(m2D.ξ/1e3, abs_err)
    ax[2].annotate(@sprintf("Max = %1.1e Sv", maximum(abs_err)), (0.05, 0.5), xycoords="axes fraction")
    ax[2].set_xlim([0, 5e3])
    savefig("images/psi_error.png")
    println("images/psi_error.png")
    plt.close()

    return Ψ2D, Ψ3D
end

function plot_convergence()
   fig, ax = subplots() 
   ax.set_title(L"Flat bottom, Bump function at $x = 0$")
   ax.set_xlabel("Resolution (km)")
   ax.set_ylabel("Maximum absolute error (Sv)")
   ax.loglog([79, 53, 26], [2.8e-3, 1.5e-3, 2.3e-4], "o", label="Linear")
   ax.loglog([66, 40, 26], [7.0e-4, 9.7e-5, 1.0e-4], "o", label="Quadratic")
   ax.loglog([60, 40], [9e-4, (40/60)^2*9e-4], "k--")
   ax.loglog([60, 40], [7e-4, (40/60)^4*7e-4], "k--")
   ax.annotate(L"$h^2$", (50, 8e-4))
   ax.annotate(L"$h^4$", (50, 2e-4))
   ax.set_xlim([20, 90])
   ax.set_ylim([5e-5, 5e-3])
   ax.legend(frameon=true, fancybox=false)
   savefig("images/convergence.png")
   println("images/convergence.png")
end

function baroclinic_convergence_1D()
    # params
    ρ₀ = 1000.
    nσ = 2^8
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  
    ν = 1e-3*ones(nσ)
    f = 1e-4
    H = 1e0

    # numerical solution
    baroclinic_LHS = get_baroclinic_LHS(ρ₀, ν, f, H, σ)
    baroclinic_RHS = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 0)
    sol = baroclinic_LHS\baroclinic_RHS
    imap = reshape(1:2*nσ, (2, nσ)) 
    τξ = sol[imap[1, :]]
    τη = sol[imap[2, :]]

    # analytical solution (assuming b = τ₀ = 0)
    q = sqrt(f/2/ν[1])
    Hq = H*q
    z = σ*H
    denom = 1 + exp(-4*Hq) - 2Hq + 2*exp(-4*Hq)*Hq + 2*Hq^2 + 2*exp(-4*Hq)*Hq^2 + 2*exp(-2*Hq)*(2*Hq^2 - 1)cos(2*Hq) - 4*exp(-2*Hq)*Hq*sin(2*Hq)
    c1 = -2*q^2*ν[1]*ρ₀*((exp(-3*Hq) + exp(-Hq))*Hq*cos(Hq) - ((1 + Hq)*exp(-3*Hq) - exp(-Hq)*(Hq - 1))*sin(Hq)) / denom
    c2 =  2*q^2*ν[1]*ρ₀*(((1 + Hq)*exp(-3*Hq) + exp(-Hq)*(Hq - 1))*cos(Hq) - (exp(-Hq) - exp(-3*Hq))*Hq*sin(Hq)) / denom
    c3 =  2*q^2*ν[1]*ρ₀*(Hq + exp(-2*Hq)*Hq*cos(2*Hq) - exp(-2*Hq)*(1 + Hq)*sin(2*Hq)) / denom
    c4 =  2*q^2*ν[1]*ρ₀*((Hq - 1) + exp(-2*Hq)*(1 + Hq)*cos(2*Hq) + exp(-2*Hq)*Hq*sin(2*Hq)) / denom
    
    # add to array
    τξ_a = @. exp(q*z)*(c1*cos(q*z) + c2*sin(q*z)) + exp(-q*(z + H))*(c3*cos(q*(z + H)) + c4*sin(q*(z + H)))
    τη_a = @. exp(q*z)*(c1*sin(q*z) - c2*cos(q*z)) + exp(-q*(z + H))*(c4*cos(q*(z + H)) - c3*sin(q*(z + H)))

    # compare 
    abs_err = abs.(τξ - τξ_a)
    println(@sprintf("Max Abs Error: %1.1e kg m-3 s-1 (i = %d / %d)", maximum(abs_err), argmax(abs_err), nσ))
    println(@sprintf("Max τ:         %1.1e kg m-3 s-1", maximum(abs.(τξ_a))))

    # plot
    fig, ax = subplots()
    ax.set_xlabel(L"Stress $\tau H^2$ (kg m$^{-1}$ s$^{-1}$)")
    ax.set_ylabel(L"Vertical coordinate $z$ (m)")
    ax.plot(τξ*H^2,   z, label=L"$\tau^\xi$")
    ax.plot(τη*H^2,   z, label=L"$\tau^\eta$")
    ax.plot(τξ_a*H^2, z, "k--", lw=0.5, label="Analytical")
    ax.plot(τη_a*H^2, z, "k--", lw=0.5)
    ax.legend()
    # ax.set_ylim([-1, -0.9])
    savefig("images/tau_error.png")
    println("images/tau_error.png")
end

function baroclinic_convergence_full()
    # ref density
    ρ₀ = 1000.

    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)

    # vertical coordinate
    nσ = 2^8
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  

    # coriolis parameter f = f₀ + βη
    f₀ = 1e-4
    β = 0.

    # viscosity
    ν = 1e-3*ones(np, nσ)

    # baroclinic LHS matrices
    baroclinic_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU}(undef, np) 
    @showprogress "Calculating baroclinic LHSs..." for i=1:np 
        baroclinic_LHSs[i] = get_baroclinic_LHS(ρ₀, ν[i, :], f₀ + β*η[i], H[i], σ)
    end  

    # buoyancy field
    b = zeros(np, nσ)
    for j=1:nσ
        b[:, j] .= 1e-6*H*σ[j] + 0.1*1e-6*H*exp(-(σ[j] + 1)/0.1)
    end

    # buoyancy gradients
    M = nuPGCM.get_M(p, t, C₀)
    M_LU = lu(M)
    Cξ, Cη = nuPGCM.get_Cξ_Cη(p, t, C₀)
    b_x = M_LU\(Cξ*b)
    b_y = M_LU\(Cη*b)
    for i=1:np
        b_x[i, :] += -σ*Hx[i].*differentiate(b[i, :], σ)/H[i] 
        b_y[i, :] += -σ*Hy[i].*differentiate(b[i, :], σ)/H[i]
    end

    # stress due to buoyancy gradients
    baroclinic_RHSs_b = zeros(np, 2*nσ)
    for i=1:np
        coeff = ρ₀*ν[i, :]./(f₀ .+ β*η[i])
        baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(coeff.*b_x[i, :], coeff.*b_y[i, :], 0, 0, 0, 0)
    end
    τξ_b, τη_b = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs_b)

    # plot
    plot_horizontal(p, t, τξ_b[:, 1]; clabel=L"Buoyancy bottom stress $\tau^\xi_b$ (kg m$^{-1}$ s$^{-2}$)")
    savefig("images/tau_xi_b_pointwise.png")
    println("images/tau_xi_b_pointwise.png")
    plt.close()
    plot_horizontal(p, t, τη_b[:, 1]; clabel=L"Buoyancy bottom stress $\tau^\eta_b$ (kg m$^{-1}$ s$^{-2}$)")
    savefig("images/tau_eta_b_pointwise.png")
    println("images/tau_eta_b_pointwise.png")
    plt.close()
end

# baroclinic_convergence_1D()
# baroclinic_convergence_full()

# m3D = setup_model(; plots=false)
s3D = invert3D(m3D)
# plot_uξ_uη_slice(m3D, s3D)
Ψ2D, Ψ3D = plot_Ψ_error()

# plot_convergence()

println("Done.")

# res (km) | max abs err (Sv)

### flat bottom, bump in b
## linear
# 79: 2.4293494638453e-03 (2.4293494638451e-03 "pointwise" b gradients)
# 53: 1.3e-3 (1.3e-3)
# 26: 2.0e-4 (2.0e-4)
## quad
# 66: 6.2e-4 (6.2e-4)
# 40: 8.5e-5 (8.5e-5)
# 26: 9.0e-5 (9.0e-5)

### bowl (5 m) (all pointwise)
## linear
# 79: 5.7e-3
# 53: 2.0e-3
# 26: 1.1e-3 
## quad
# 66: 2.2e-3 
# 40: 1.4e-3

### bowl (5 m) (all global)
## linear
# 79: 3.2e-2
# 53: 9.1e-3
## quad
# 66: 3.8e-1 
# 40: 2.3e-3