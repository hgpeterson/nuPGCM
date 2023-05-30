using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
pygui(false)

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

# beta_sim_setup()
overturn_sim_setup()