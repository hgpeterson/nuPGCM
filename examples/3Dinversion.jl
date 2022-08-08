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
    # bath = "flat"
    bath = "tub"
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

    # depth scale
    H₀ = 2e3

    # gaussian 
    Δ = Lx/5 
    G(r) = 1 - exp(-r^2/(2*Δ^2)) 
    Gr(r) = r/Δ^2*exp(-r^2/(2*Δ^2))

    # bump function
    w = 4*Δ
    c = 0
    G_bump(r) = if c - w < r < c + w return exp(1 - w^2/(w^2 - (r - c)^2)) else return 0 end 
    Gr_bump(r) = -2*(r - c)*w^2*G_bump(r)/(w^2 - (r - c)^2)^2

    # calculate H(x, y)
    if bath == "flat"
        # flat bottom
        H = H₀*ones(np)
        Hx = zeros(np)
        Hy = zeros(np)
    elseif bath == "tub"
        if geo == "square"
            # square bathtub
            H  = @. H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) + 100
            Hx = @. H₀*Gr(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gr(Lx - ξ)*G(Ly + η)*G(Ly - η)
            Hy = @. H₀*G(Lx + ξ)*G(Lx - ξ)*Gr(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gr(Ly - η)
        elseif geo == "circle"
            # circular bathtub (radius = Lx)
            H  = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + eps()
            Hx = @. H₀*Gr(sqrt(ξ^2 + η^2) - Lx)*ξ/sqrt(ξ^2 + η^2)
            Hy = @. H₀*Gr(sqrt(ξ^2 + η^2) - Lx)*η/sqrt(ξ^2 + η^2)
        end
    elseif bath == "bump"
        if geo == "circle"
            # circular bump
            H  = @. H₀ - 2e2*G_bump(sqrt(ξ^2 + η^2))
            Hx = @.    - 2e2*Gr_bump(sqrt(ξ^2 + η^2))*ξ/sqrt(ξ^2 + η^2)
            Hy = @.    - 2e2*Gr_bump(sqrt(ξ^2 + η^2))*η/sqrt(ξ^2 + η^2)
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
    N² = m.N²[1, 1] # constant 
    for j=1:m.nσ
        b[:, j] .= N²*m.H*(m.σ[j] + 0.1*exp(-(m.σ[j] + 1)/0.1))
        # b[:, j] .= N²*m.H*m.σ[j] 
    end
    # for i=1:np
    #     Δ = 0.9*m.Lx
    #     if sqrt(ξ[i]^2 + η[i]^2) < Δ
    #         b[i, :] .= m.N²[i, :]*m.H[i].*m.σ * (1 - 0.1*exp(-Δ^2/(Δ^2 - ξ[i]^2 - η[i]^2)))
    #     else
    #         b[i, :] .= m.N²[i, :]*m.H[i].*m.σ
    #     end
    # end
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

    # # analytical buoyancy gradients
    # rhs_x = zeros(np, m.nσ)
    # rhs_y = zeros(np, m.nσ)
    # for j=1:m.nσ
    #     # bξ = (0.1 + m.σ[j])*N²*m.Hx*exp(-(m.σ[j] + 1)/0.1)
    #     # bη = (0.1 + m.σ[j])*N²*m.Hy*exp(-(m.σ[j] + 1)/0.1)
    #     bξ = m.Hx./m.H.*b[:, j]
    #     bη = m.Hy./m.H.*b[:, j]
    #     bσ = N²*m.H*(1 - exp(-(m.σ[j] + 1)/0.1))
    #     bx = bξ - m.σ[j]*m.Hx./m.H.*bσ
    #     by = bη - m.σ[j]*m.Hy./m.H.*bσ
    #     rhs_x[:, j] = m.ρ₀*m.ν[:, j]./(m.f₀ .+ m.β*η).*bx
    #     rhs_y[:, j] = m.ρ₀*m.ν[:, j]./(m.f₀ .+ m.β*η).*by
    # end
    # baroclinic_RHSs_b = zeros(m.np, 2*m.nσ)
    # for i=1:np
    #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
    # end
    # τξ_b, τη_b = get_τξ_τη(m.baroclinic_LHSs, baroclinic_RHSs_b)

    # integrals of buoyancy gradients on rhs
    bσ_x = zeros(np, m.nσ)
    bσ_y = zeros(np, m.nσ)
    for i=1:np
        bσ_x[i, :] = -m.σ*Hx[i]/H[i].*differentiate(b[i, :], m.σ) 
        bσ_y[i, :] = -m.σ*Hy[i]/H[i].*differentiate(b[i, :], m.σ)
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

    # # pointwise buoyancy gradients
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
    println(@sprintf("Max Abs. Err.: %1.1e Sv (%d km)", maximum(abs_err), m2D.ξ[argmax(abs_err)]/1e3))
    println(@sprintf("Max Abs. Ψ:    %1.1e Sv (%d km)", maximum(abs.(Ψ2D))/1e6, m2D.ξ[argmax(Ψ2D)]/1e3))

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

m3D = setup_model()
# m3D = setup_model(; plots=false)
# s3D = invert3D(m3D)
# Ψ2D, Ψ3D = plot_Ψ_error()
# plot_uξ_uη_slice(m3D, s3D)

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

### bump in H
## linear
# 79: 1.5e-3
# 53: 7.0e-4
# 26: 4.6e-4
## quad
# 66: 8.6e-4
# 40: 4.9e-4

### bowl (all pointwise bx, by)
## linear
# 79: 9.7e-3
# 53: 3.7e-3
# 26: 4.8e-4

### bowl (all global bx, by)
## linear
# 79: 5.7e-3
# 53: 2.0e-3
# 26: 1.1e-3

### bowl (analytical bx, by)
## linear
# 79: 1.1e-2
# 53: 4.8e-3 
# 26: 9.2e-4
## quad
# 66: 1.8e-3
# 40: 7.6e-4
# 26: 4.1e-4