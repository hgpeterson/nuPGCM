using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SuiteSparse
using ProgressMeter
using Dierckx

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_basin_geometry()
    # geometry type
    # geo = "square"
    geo = "circle"

    # bathymetry type
    # bath = "flat"
    # bath = "tub"
    bath = "bump"

    # resolution
    # res = 1   #  1452 linear nodes,   5677 quadratic nodes
    # res = 2   #  4027 linear nodes,  15899 quadratic nodes
    # res = 3   #  9062 linear nodes,  35936 quadratic nodes
    # res = 4   # 16114 linear nodes
    res = 5   # 36268 linear nodes, 144433 quadratic nodes

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
            H  = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + 100
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
    # f₀ = 0.
    # β = 2e-11

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
        plot_horizontal(p, t, m.τξ_tξ[:, 1]; clabel=L"Symmetric bottom stress $\tau^\xi_{t\xi}$ (kg m$^{-1}$ s$^{-1}$)", contours=false)
        savefig("images/tau_xi_t.png")
        println("images/tau_xi_t.png")
        plt.close()
        plot_horizontal(p, t, m.τη_tξ[:, 1]; clabel=L"Anti-symmetric bottom stress $\tau^\eta_{t\xi}$ (kg m$^{-1}$ s$^{-1}$)", contours=false)
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
    end

    # wind stress
    # τξ₀ = @. -0.1*cos(π*η/Ly)
    τξ₀ = zeros(m.np)
    τη₀ = zeros(m.np)

    # invert
    Ψ, Huξ, Huη, Huσ = invert(m, τξ₀, τη₀, b; plots=true)

    # save state
    s = ModelState3DPG(b, Ψ, Huξ, Huη, Huσ, [1])

    return s
end

function plot_uξ_uη_slice(m, s)
    ξ_slice = (-m.Lx + 1e4):m.Lx/2^7:(m.Lx - 1e4)
    η₀ = 0
    # ξ₀ = 0
    # η_slice = (-m.Ly + 1e4):m.Ly/2^7:(m.Ly - 1e4)

    # plot uξ slice
    ax = plot_ξ_slice(m, s, s.uξ./m.H, ξ_slice, η₀; clabel=L"Zonal velocity $u^x$ (m s$^{-1}$)", contours=false)
    # ax = plot_η_slice(m, s, s.uξ, η_slice, ξ₀; clabel=L"Zonal velocity $u^x$ (m s$^{-1}$)", contours=false)
    ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    ax.set_xticks(-m.Lx/1e3:2500:m.Lx/1e3)
    ax.set_ylim([-maximum(m.H)/1e3, 0])
    savefig("images/ux3D.png")
    println("images/ux3D.png")
    plt.close()

    # plot uη slice
    ax = plot_ξ_slice(m, s, s.uη./m.H, ξ_slice, η₀; clabel=L"Meridional velocity $u^y$ (m s$^{-1}$)", contours=false)
    # ax = plot_η_slice(m, s, s.uη, η_slice, ξ₀; clabel=L"Meridional velocity $u^y$ (m s$^{-1}$)", contours=false)
    ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    ax.set_xticks(-m.Lx/1e3:2500:m.Lx/1e3)
    ax.set_ylim([-maximum(m.H)/1e3, 0])
    savefig("images/uy3D.png")
    println("images/uy3D.png")
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

function print_u_error()
    # load 2D
    m2D = load_setup_2D("../output/setup2D.h5")
    s2D = load_state_2D("../output/state2D.h5")

    # compute error between 2D and 3D at each ξ point on the 2D grid and σ point on the 3D grid
    abs_err_uξ = zeros(m2D.nξ, m2D.nσ)
    abs_err_uη = zeros(m2D.nξ, m2D.nσ)
    abs_err_uσ = zeros(m2D.nξ, m2D.nσ)
    @showprogress "Evaluating errors..." for i=1:m2D.nξ-1
        # interpolate 2D in σ
        uξ2D = Spline1D(m2D.σ, s2D.uξ[i, :])
        uη2D = Spline1D(m2D.σ, s2D.uη[i, :])
        uσ2D = Spline1D(m2D.σ, s2D.uσ[i, :])

        for j=1:m3D.nσ
            # compute 3D u at (m2D.ξ[i], m3D.σ[j])
            # uξ3D = fem_evaluate(m3D, s3D.uξ[:, j], m2D.ξ[i], 0)
            # uη3D = fem_evaluate(m3D, s3D.uη[:, j], m2D.ξ[i], 0)
            # uσ3D = fem_evaluate(m3D, s3D.uσ[:, j], m2D.ξ[i], 0)
            H = fem_evaluate(m3D, m3D.H, m2D.ξ[i], 0)
            uξ3D = fem_evaluate(m3D, s3D.uξ[:, j], m2D.ξ[i], 0)/H
            uη3D = fem_evaluate(m3D, s3D.uη[:, j], m2D.ξ[i], 0)/H
            uσ3D = fem_evaluate(m3D, s3D.uσ[:, j], m2D.ξ[i], 0)/H

            # evaluate 2D interpolation at m3D.σ[j], save error
            abs_err_uξ[i, j] = abs(uξ3D - uξ2D(m3D.σ[j]))
            abs_err_uη[i, j] = abs(uη3D - uη2D(m3D.σ[j]))
            abs_err_uσ[i, j] = abs(uσ3D - uσ2D(m3D.σ[j]))
        end
    end

    # print results
    println(@sprintf("%d km", m3D.Lx/sqrt(m3D.np)/1e3))
    println(@sprintf("Max Err. uξ: %1.1e m s⁻¹ (%d km)", maximum(abs_err_uξ),   m2D.ξ[argmax(abs.(abs_err_uξ))[1]]/1e3))
    println(@sprintf("Max uξ:      %1.1e m s⁻¹ (%d km)", maximum(abs.(s2D.uξ)), m2D.ξ[argmax(abs.(s2D.uξ))[1]]/1e3))
    println(@sprintf("Max Err. uη: %1.1e m s⁻¹ (%d km)", maximum(abs_err_uη),   m2D.ξ[argmax(abs.(abs_err_uη))[1]]/1e3))
    println(@sprintf("Max uη:      %1.1e m s⁻¹ (%d km)", maximum(abs.(s2D.uη)), m2D.ξ[argmax(abs.(s2D.uη))[1]]/1e3))
    println(@sprintf("Max Err. uσ: %1.1e m s⁻¹ (%d km)", maximum(abs_err_uσ),   m2D.ξ[argmax(abs.(abs_err_uσ))[1]]/1e3))
    println(@sprintf("Max uσ:      %1.1e m s⁻¹ (%d km)", maximum(abs.(s2D.uσ)), m2D.ξ[argmax(abs.(s2D.uσ))[1]]/1e3))
end

# m3D = setup_model()
m3D = setup_model(; plots=false)
s3D = quick_invert(m3D)
Ψ2D, Ψ3D = plot_Ψ_error()
# print_u_error()
# plot_uξ_uη_slice(m3D, s3D)

println("Done.")


## tub

# H: 0.01, ν: 1e-3, nσ: 2^7
# 53: 2.6e-3
# 26: 1.5e-3

# H: 0.01, ν: 1e-1, nσ: 2^7
# 53: 2.0e-3
# 26: 1.2e-3

# H: 100, ν: 1e-1, nσ: 2^7
# 53: 2.2e-3 
# 26: 1.5e-3

## bump

# H: 200, ν: 1e-1
# 53: 6.9e-4
# 26: 4.6e-4