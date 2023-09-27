using nuPGCM
using Printf
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/bowl2D/")

function run()
    # parameters
    ε² = 1e-2
    μ = 1e0
    ϱ = 1e0
    α = ε²/μ/ϱ
    T = 5e-2/α
    n_steps = 500
    Δt = T/n_steps
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
    # κ_func(ξ, σ) = ε²/μ/ϱ
    κ_func(ξ, σ) = ε²/μ/ϱ*(1e-2 + exp(-H_func(ξ)*(σ + 1)/0.1))

    # viscosity
    ν_func(ξ, σ) = μ*ϱ*κ_func(ξ, σ)

    # stratification
    N2 = 1
    N2_func(ξ, σ) = N2
    
    # timestepping
    t_plot = T/5
    t_save = T/5
    
    # create model struct
    m = ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup(m)

    # set initial state
    b = copy(m.z)
    # δ = 0.2
    # b = @. m.z + δ*exp(-(m.z+m.H)/δ) - δ*exp(m.z/δ)
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2D(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, T, t_plot, t_save) 

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
    savefig("$(out_folder)ux_bowl2D.png")
    println("$(out_folder)ux_bowl2D.png")
    plt.close()

    fig, ax = plt.subplots(1)
    field = s.uη
    vmax = maximum(abs.(field))
    vmin = -vmax
    img = ax.pcolormesh(m.x, m.z, field, cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    # levels = range(-vmax, vmax, length=8)
    # ax.contour(m.x, m.z, field, levels=levels, colors="k", linestyles="-", linewidths=0.25)
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
    savefig("$(out_folder)uy_bowl2D.png")
    println("$(out_folder)uy_bowl2D.png")
    plt.close()

    ix = argmin(abs.(m.ξ .- 0.5))
    H = m.H[ix]
    z = m.z[ix, :]
    ωx = -1/H*differentiate(s.uη[ix, :], m.σ)
    ωy =  1/H*differentiate(s.uξ[ix, :], m.σ)
    χx =  H*cumtrapz(s.uη[ix, :], m.σ)
    χy = -H*cumtrapz(s.uξ[ix, :], m.σ)
    b = s.b[ix, :]
    bz = 1/H*differentiate(s.b[ix, :], m.σ)
    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)
    ax[1, 1].plot(ωx, z)
    ax[1, 2].plot(ωy, z)
    ax[1, 3].plot(b, z)
    ax[2, 1].plot(χx, z)
    ax[2, 2].plot(χy, z)
    ax[2, 3].plot(bz, z)
    ax[1, 1].set_xlabel(L"\omega^x")
    ax[1, 2].set_xlabel(L"\omega^y")
    ax[1, 3].set_xlabel(L"b")
    ax[2, 1].set_xlabel(L"\chi^x")
    ax[2, 2].set_xlabel(L"\chi^y")
    ax[2, 3].set_xlabel(L"\partial_z b")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 1].set_ylim(-H, 0)
    ax[2, 1].set_ylim(-H, 0)
    ax[1, 2].set_title(latexstring(@sprintf("2D Profiles at \$x = %1.1f\$", 0.5)))
    for a ∈ ax
        a.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    end
    savefig("$(out_folder)profiles2D.png")
    println("$(out_folder)profiles2D.png")
    plt.close()

    Uy = [trapz(s.uη[i, :], m.z[i, :]) for i=1:m.nξ]
    Ψ = cumtrapz(Uy, m.x[:, 1]) .- trapz(Uy, m.x[:, 1]) 
    fig, ax = plt.subplots(1)
    ax.plot(m.x[:, 1], Ψ)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Barotropic streamfunction $\Psi$")
    savefig("$(out_folder)psi_bowl2D.png")
    println("$(out_folder)psi_bowl2D.png")
    plt.close()

    return m, s
end

m2D, s2D = run()

println("Done.")