using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output/bowl2D/")

function run()
    # parameters
    ε² = 1e-2
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 1.
    L = 1.
    nξ = 2^8 
    nσ = 2^8
    coords = "axisymmetric"
    periodic = false
    bl = false

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L)
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: bowl
    no_net_transport = true
    H0 = 1
    H_func(x) = H0*(1 - (x/L)^2) + 0.005*H0
    Hx_func(x) = -2*H0*x/L^2

    # diffusivity
    κ_func(ξ, σ) = ε²/μ/ϱ

    # viscosity
    ν_func(ξ, σ) = μ*ϱ*κ_func(ξ, σ)

    # stratification
    N2 = 1
    N2_func(ξ, σ) = N2
    
    # timestepping
    t_plot = 10*Δt
    t_save = 10*Δt
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup(m)

    # set initial state
    b = copy(m.z)
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, 5*t_save, t_plot, t_save) 

    # plots
    fig, ax = plt.subplots(1)
    field = s.uξ
    vmax = maximum(abs.(field))
    vmin = -vmax
    img = ax.pcolormesh(m.x, m.z, field, cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    cb = colorbar(img, ax=ax, label=L"Cross-slope flow $u^x$")
    n_levels = 20
    iξ = argmax(m.H)
    lower_level = -trapz(m.N2[iξ, :], m.z[iξ, :])
    upper_level = lower_level/100
    levels = lower_level:(upper_level - lower_level)/(n_levels - 1):upper_level
    ax.contour(m.x, m.z, s.b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
    ax.fill_between(m.x[:, 1], m.z[:, 1], minimum(m.z), color="k", alpha=0.3, lw=0.0)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.set_xlim([0, m.ξ[end]])
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig("$out_folder/ux_bowl2D.png")
    println("$out_folder/ux_bowl2D.png")
    plt.close()

    fig, ax = plt.subplots(1)
    field = s.uη
    vmax = maximum(abs.(field))
    vmin = -vmax
    img = ax.pcolormesh(m.x, m.z, field, cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    cb = colorbar(img, ax=ax, label=L"Along-slope flow $u^y$")
    n_levels = 20
    iξ = argmax(m.H)
    lower_level = -trapz(m.N2[iξ, :], m.z[iξ, :])
    upper_level = lower_level/100
    levels = lower_level:(upper_level - lower_level)/(n_levels - 1):upper_level
    ax.contour(m.x, m.z, s.b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
    ax.fill_between(m.x[:, 1], m.z[:, 1], minimum(m.z), color="k", alpha=0.3, lw=0.0)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.set_xlim([0, m.ξ[end]])
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig("$out_folder/uy_bowl2D.png")
    println("$out_folder/uy_bowl2D.png")
    plt.close()

    Uy = [trapz(s.uη[i, :], m.z[i, :]) for i=1:m.nξ]
    Ψ = cumtrapz(Uy, m.x[:, 1]) .- trapz(Uy, m.x[:, 1]) 
    fig, ax = plt.subplots(1)
    ax.plot(m.x[:, 1], Ψ)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Barotropic streamfunction $\Psi$")
    savefig("$out_folder/psi_bowl2D.png")
    println("$out_folder/psi_bowl2D.png")
    plt.close()
end

run()

println("Done.")