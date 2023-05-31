using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
pygui(false)
plt.close()

function beta_sim_setup()
    g = FEGrid(1, "meshes/circle/mesh2.h5")
    H(x) = 1 - x[1]^2 - x[2]^2
    u = [H(g.p[i, :]) for i=1:g.np]
    fig, ax = plt.subplots(1)
    im = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u, cmap="Blues", vmin=0, vmax=1, rasterized=true, edgecolors="k", linewidth=0.1)
    levels = [1/4, 1/2, 3/4]
    ax.tricontour(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u, colors="k", linewidths=0.5, linestyles="-", levels=levels)
    cb = colorbar(im, ax=ax, label=L"Depth $H$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    plt.axis("equal")
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Meridional Coordinate $y$")
    ax.set_yticks(-1:0.5:1)
    savefig("beta_sim_setup.pdf")
    println("beta_sim_setup.pdf")
    plt.close()
end

function overturn_sim_setup()
    g = FEGrid(1, "meshes/square/mesh3.h5")
    g.p[:, 1] = (g.p[:, 1] .+ 1)/2
    Δ = 0.1
    G(r) = 1 - exp(-r^2/(2Δ^2))
    sigmoid(r) = 1/(1 + exp(-r/Δ))
    H(x) = (G(x[1]) + sigmoid(-0.5-x[2])*(1 - G(x[1])))*(G(1 - x[1]) + sigmoid(-0.5-x[2])*(1 - G(1 - x[1])))*G(1 - x[2])*G(1 + x[2]) #+ sigmoid(-0.75 - x[2])
    u = [H(g.p[i, :]) for i=1:g.np]
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    im = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u, cmap="Blues", vmin=0, vmax=1, rasterized=true, shading="gouraud")
    levels = [1/4, 1/2, 3/4]
    ax.tricontour(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u, colors="k", linewidths=0.5, linestyles="-", levels=levels)
    cb = colorbar(im, ax=ax, label=L"Depth $H$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.annotate("", xy=(0.2, 0.16), xytext=(-0.1, 0.16), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    ax.annotate("", xy=(1.08, 0.16), xytext=(0.78, 0.16), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    plt.axis("equal")
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Meridional Coordinate $y$")
    savefig("overturn_sim_setup.pdf")
    println("overturn_sim_setup.pdf")
    plt.close()
end

function llc4320res()
     delR = [1.00,    1.14,    1.30,    1.49,   1.70,
          1.93,    2.20,    2.50,    2.84,   3.21,
          3.63,    4.10,    4.61,    5.18,   5.79,
          6.47,    7.20,    7.98,    8.83,   9.73,
         10.69,   11.70,   12.76,   13.87,  15.03,
         16.22,   17.45,   18.70,   19.97,  21.27,
         22.56,   23.87,   25.17,   26.46,  27.74,
         29.00,   30.24,   31.45,   32.65,  33.82,
         34.97,   36.09,   37.20,   38.29,  39.37,
         40.45,   41.53,   42.62,   43.73,  44.87,
         46.05,   47.28,   48.56,   49.93,  51.38,
         52.93,   54.61,   56.42,   58.38,  60.53,
         62.87,   65.43,   68.24,   71.33,  74.73,
         78.47,   82.61,   87.17,   92.21,  97.79,
        103.96,  110.79,  118.35,  126.73, 136.01,
        146.30,  157.71,  170.35,  184.37, 199.89,
        217.09,  236.13,  257.21,  280.50, 306.24,
        334.64,  365.93,  400.38,  438.23, 479.74]
    z = cumsum(-delR)
    z = z[-2000 .> z .> -5000]
    plt.plot(zeros(size(z, 1)), z, "ro", ms=1)
    plt.ylim(-5100, -1900)
    savefig("plots/llc4320res.svg")
    println("plots/llc4320res.svg")
end

function uy_bowl2D()
    # parameters
    f = 1e0
    L = 1e0
    nξ = 2^8 
    nσ = 2^8
    coords = "axisymmetric"
    periodic = false
    bl = false
    out_folder = "output/"

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L)
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: bowl
    no_net_transport = true
    H0 = 1e0
    H_func(x) = H0*(1 - (x/L)^2) + 0.005*H0
    Hx_func(x) = -2*H0*x/L^2

    # diffusivity
    κ0 = 1e-4
    κ1 = 0
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    μ = 1.
    ν_func(ξ, σ) = μ*κ_func(ξ, σ)

    # stratification
    N2 = 1
    N2_func(ξ, σ) = N2
    
    # timestepping
    Δt = 1e-1
    t_plot = 20*Δt
    t_save = 20*Δt
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup(m)

    # set initial state
    b = zeros(nξ, nσ)
    δ = 0.1*H0
    for i=1:nξ
        # b[i, :] = cumtrapz(m.N2[i, :], m.z[i, :]) .- trapz(m.N2[i, :], m.z[i, :])
        b[i, :] = @. m.N2[i, :]*(m.z[i, :] + δ*exp(-(m.z[i, :] + m.H[i])/δ))
    end
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # # solve
    # evolve!(m, s, 100*Δt, t_plot, t_save) 

    fig, ax = plt.subplots(1)
    field = s.uη
    vmax = maximum(abs.(field))
    vmin = -vmax
    img = ax.pcolormesh(m.x, m.z, field, cmap="RdBu_r", vmin=-0.15, vmax=0.15, rasterized=true, shading="auto")
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
    # ax.set_xlim([m.ξ[1]/1e3, (m.ξ[end] + m.ξ[2] - m.ξ[1])/1e3])
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    # ridge_plot(m, s, s.uη, L"t = 30", L"Along-slope flow $u^y$"; style="pcolormesh")
    savefig("plots/uy_bowl2D.pdf")
    println("plots/uy_bowl2D.pdf")
    plt.close()

    # Uy = [trapz(s.uη[i, :], m.z[i, :]) for i=1:m.nξ]
    # Ψ = -(trapz(Uy, m.x[:, 1]) .- cumtrapz(Uy, m.x[:, 1]))
    # plt.plot(m.x[:, 1], Ψ)
    # savefig("plots/Psi.png")
    # println("plots/Psi.png")
    # plt.close()
end

# beta_sim_setup()
# overturn_sim_setup()
# llc4320res()
uy_bowl2D()