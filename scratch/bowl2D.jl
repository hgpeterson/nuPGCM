using nuPGCM
using Printf
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/bowl2D/")

function run()
    # parameters
    ε² = 1e-4
    μϱ = 1e0
    T = 1e1
    Δt = 1e-2
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
    H_func(x) = H0*(1 - (x/L)^2) + 0.0001*H0
    Hx_func(x) = -2*H0*x/L^2

    # diffusivity
    # κ_func(ξ, σ) = ε²/μ/ϱ
    κ_func(ξ, σ) = ε²/μϱ*(1e-2 + exp(-H_func(ξ)*(σ + 1)/0.1))

    # viscosity
    # ν_func(ξ, σ) = μϱ*κ_func(ξ, σ)
    ν_func(ξ, σ) = ε²

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
    # b = @. m.z + 0.1*exp(-(m.z+m.H)/0.1) #- 0.1*exp(m.z/0.1)
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2D(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, T, t_plot, t_save) 

    # cartesian
    ux, uy, uz = transform_from_TF(m, s)
    ωx = zeros(m.nξ, m.nσ)
    ωy = zeros(m.nξ, m.nσ)
    for i ∈ 1:m.nξ
        ωx[i, :] = -differentiate(uy[i, :], m.z[i, :])
        ωy[i, :] = +differentiate(ux[i, :], m.z[i, :])
    end

    # integrals
    @printf("b_prod  = % 1.5e\n", integrate2D(m, uz.*s.b))
    @printf("ke_diss = % 1.5e\n", integrate2D(m, m.ν.*(ωx.^2 .+ ωy.^2)))

    # slices
    plot2D(m, s, ux, cb_label=L"Zonal velocity $u^x$", filename="$(out_folder)ux_bowl2D.png")
    plot2D(m, s, uy, cb_label=L"Meridional velocity $u^y$", filename="$(out_folder)uy_bowl2D.png")
    plot2D(m, s, uz, cb_label=L"Vertical velocity $u^z$", filename="$(out_folder)uz_bowl2D.png")
    plot2D(m, s, uz.*s.b, cb_label=L"Buoyancy production $u^z b$", filename="$(out_folder)uzb_bowl2D.png")

    # profiles
    ix = argmin(abs.(m.ξ .- 0.5))
    H = m.H[ix]
    z = m.z[ix, :]
    ωx = -1/H*differentiate(s.uη[ix, :], m.σ)
    ωy =  1/H*differentiate(s.uξ[ix, :], m.σ)
    χx =  H*cumtrapz(s.uη[ix, :], m.σ)
    χy = -H*cumtrapz(s.uξ[ix, :], m.σ)
    ux, uy, uz = transform_from_TF(m, s)
    ux = ux[ix, :]
    uy = uy[ix, :]
    uz = uz[ix, :]
    b = s.b[ix, :]
    bz = 1/H*differentiate(s.b[ix, :], m.σ)
    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)
    ax[1, 1].plot(ωx, z, label=L"\omega^x")
    ax[1, 1].plot(ωy, z, label=L"\omega^y")
    ax[1, 2].plot(χx, z, label=L"\chi^x")
    ax[1, 2].plot(χy, z, label=L"\chi^y")
    ax[1, 3].plot(bz, z)
    ax[2, 1].plot(ux, z)
    ax[2, 2].plot(uy, z)
    ax[2, 3].plot(uz, z)
    ax[1, 1].set_xlabel("Vorticity")
    ax[1, 2].set_xlabel("Streamfunction")
    ax[1, 3].set_xlabel(L"Stratification $\partial_z b$")
    ax[2, 1].set_xlabel(L"Zonal flow $u^x$")
    ax[2, 2].set_xlabel(L"Meridional flow $u^y$")
    ax[2, 3].set_xlabel(L"Vertical flow $u^z$")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 1].set_ylim(-H, 0)
    ax[2, 1].set_ylim(-H, 0)
    ax[1, 1].legend()
    ax[1, 2].legend()
    for a ∈ ax
        a.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    end
    ax[1, 2].set_title(latexstring(@sprintf("2D Profiles at \$x = %1.1f\$", 0.5)))
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

function integrate2D(m, field)
    xint = 2π*[trapz(m.ξ[i]*field[i, :], m.z[i, :]) for i=1:m.nξ]
    return trapz(xint, m.ξ)
end

function plot2D(m::ModelSetup2D, s::ModelState2D, field; cb_label, filename)
    fig, ax = plt.subplots(1)
    vmax = maximum(abs.(field))
    vmin = -vmax
    img = ax.pcolormesh(m.x, m.z, field, cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    colorbar(img, ax=ax, label=cb_label)
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
    savefig(filename)
    println(filename)
    plt.close()
end

# m2D, s2D = run()

# save profile
using HDF5
u, v, w = transform_from_TF(m2D, s2D)
bz = ∂z(m2D, s2D.b)
iξ = argmin(abs.(m2D.ξ .- 0.5))
h5open("gamma0.h5", "w") do file
    write(file, "u", u[iξ, :])
    write(file, "v", v[iξ, :])
    write(file, "w", w[iξ, :])
    write(file, "bz", bz[iξ, :])
    write(file, "z", m2D.z[iξ, :])
end
println("gamma0.h5")

println("Done.")