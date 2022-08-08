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
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    μ = 1e0
    κ = zeros(np, nσ)
    for i=1:nσ
        κ[:, i] = @. κ0 + κ1*exp(-H*(σ[i] + 1)/h)
    end
    ν = μ*κ
    # ν = 1e-3*ones(np, nσ)
    # κ = 1e-3*ones(np, nσ)
    # ν = 1e-1*ones(np, nσ)
    # κ = 1e-1*ones(np, nσ)

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

function quick_invert(m)
    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # buoyancy field
    b = zeros(m.np, m.nσ)
    N² = m.N²[1, 1] # constant 
    for j=1:m.nσ
        b[:, j] .= N²*m.H*(m.σ[j] + 0.1*exp(-(m.σ[j] + 1)/0.1))
        # b[:, j] .= N²*m.H*m.σ[j] 
    end

    # wind stress
    # τξ₀ = @. -0.1*cos(π*η/Ly)
    τξ₀ = zeros(m.np)
    τη₀ = zeros(m.np)

    # invert
    Ψ, uξ, uη, uσ = invert(m, τξ₀, τη₀, b; plots=true)

    # save state
    s = ModelState3DPG(b, Ψ, uξ, uη, uσ, [1])

    # plot Ψ
    fig, ax, im = plot_horizontal(m.p, m.t, Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)")
    savefig("images/psi.png")
    println("images/psi.png")
    plt.close()

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

# m3D = setup_model()
# m3D = setup_model(; plots=false)
s3D = quick_invert(m3D)
# Ψ2D, Ψ3D = plot_Ψ_error()
# plot_uξ_uη_slice(m3D, s3D)

println("Done.")