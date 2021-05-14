# for loading data
include("1dtc_pg/utils.jl")
include("1dtc_nondim/utils.jl")
include("2dpg/utils.jl")

# matplotlib
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

function sketchRidge()
    fig, ax = subplots(1)
    ax.fill_between(x[:, 1]/1000, z[:, 1]/1000, minimum(z)/1000, color="k", alpha=0.3, lw=0.0)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_ylim([minimum(z)/1000, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    tight_layout()
    savefig("sketchRidge.svg")
    println("sketchRidge.svg")
end

function sketchSlope()
    x = 0:0.001:1
    z = 0:0.001:1
    Px = repeat(x, 1, size(z, 1))
    println(size(Px))
    fig, ax = subplots(1)
    ax.pcolormesh(x, z, Px', cmap="viridis", shading="auto", rasterized=true)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks([])
    ax.set_yticks([])
    tight_layout()
    savefig("sketchSlope.svg")
    println("sketchSlope.svg")
end

function chiForSketch(folder)
    iξ = argmin(abs.(ξ .- L/4))
    fig, ax = subplots(1, figsize=(3.404/1.62, 3.404))
    c = loadCheckpoint1DTCPG(string(folder, "1dcan/checkpoint1000.h5"))
    ax.plot(c.chi[1, :]/maximum(c.chi[1, :]), z[iξ, :], "k")
    c = loadCheckpoint1DTCPG(string(folder, "1dtc/checkpoint1000.h5"))
    ax.plot(c.chi[1, :]/maximum(c.chi[1, :]), z[iξ, :], "k")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(false)
    ax.axvline(0, ls="-", lw=0.5, c="k")

    tight_layout()
    savefig("chiForSketch.svg", transparent=true)
    println("chiForSketch.svg")
end

function chi_v_ridge(folder)
    # load
    c = loadCheckpoint2DPG(string(folder, "2dpg/Pr1/checkpoint1.h5"))
    u, v, w = transformFromTF(c.uξ, c.uη, c.uσ)

    # plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)
    ridgePlot(c.χ, c.b, "", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1])
    ridgePlot(v, c.b, "", L"along-ridge flow, $v$ (m s$^{-1}$)"; ax=ax[2])
    ax[1].plot([L/1e3/4, L/1e3/4], [-H(L/4)/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([L/1e3/4, L/1e3/4], [-H(L/4)/1e3, 0], "r-", alpha=0.5)
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")
    savefig("spinupRidge.pdf")
    println("spinupRidge.pdf")
    close()
end

function profiles2Dvs1D(folder, σ)
    ix = argmin(abs.(ξ .- L/4))
    tDays = 1000:1000:5000
    
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    fig.text(0.05, 0.98, string("Canonical 1D (Pr = ", σ, "):"), ha="left", va="top")
    fig.text(0.05, 0.52, string(L"2D $\nu$PGCM (Pr = ", σ, "):"), ha="left", va="top")

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 1].set_xlabel(string(L"streamfunction, $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow, $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification, $B_z$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(tDays, 1)))

    # fixed x
    if σ == 1
        ax[1, 1].set_xlim([0, 57])
        ax[2, 1].set_xlim([-0.1, 1.65])
        ax[1, 2].set_xlim([-2.7, 1.4])
        ax[2, 2].set_xlim([-2.7, 1.4])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    elseif σ == 200
        ax[1, 1].set_xlim([0, 190])
        ax[2, 1].set_xlim([-5, 95])
        ax[1, 2].set_xlim([-2.0, 0.3])
        ax[2, 2].set_xlim([-2.0, 0.3])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    end

    # fixed y
    ax[1, 1].set_ylim([-2, 0])
    ax[2, 1].set_ylim([-2, 0])

    # plot data from folder
    for i=1:size(tDays, 1)
        tDay = tDays[i]
        label = string(Int64(tDay), " days")
        # canonical 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dcan/Pr", σ, "/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :],     label=label)
        ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        
        # 2D PG solution
        c = loadCheckpoint2DPG(string(folder, "2dpg/Pr", σ, "/checkpoint", i, ".h5"))
        u, v, w = transformFromTF(c.uξ, c.uη, c.uσ)
        Bz = c.N^2 .+ zDerivativeTF(c.b)
        ax[2, 1].plot(1e3*c.χ[ix, :], z[ix, :]/1e3, c=colors[i, :],     label=label)
        ax[2, 2].plot(1e2*v[ix, :],   z[ix, :]/1e3, c=colors[i, :], label=label)
        ax[2, 3].plot(1e6*Bz[ix, :],  z[ix, :]/1e3, c=colors[i, :], label=label)

        # transport-constrained 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dtc/Pr", σ, "/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[2, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c="k", ls=":")
        ax[2, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c="k", ls=":")
        ax[2, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c="k", ls=":")
    end

    # steady state canonical
    c = loadCheckpoint1DTCPG(string(folder, "1dcan/Pr", σ, "/checkpoint999.h5"))
    Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
    ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")
    ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")
    ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")

    ax[2, 3].legend(loc=(0.12, 0.3))
    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["transport-\nconstrained 1D"]
    ax[2, 1].legend(custom_handles, custom_labels, loc=(0.3, 0.4))
    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1")]
    custom_labels = ["steady state"]
    ax[1, 3].legend(custom_handles, custom_labels, loc=(0.1, 0.6))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)
    savefig(string("spinupProfilesPr", σ, ".pdf"))
    println(string("spinupProfilesPr", σ, ".pdf"))
    close()
end

function spindownProfiles(folder)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    c = loadCheckpoint1DTCNondim(string(folder, "/tc/checkpoint1.h5"))
    fig.text(0.05, 0.98, string(L"Canonical 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top")
    fig.text(0.05, 0.52, string(L"Transport-Constrained 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top")

    ax[1, 1].set_ylabel(L"$\tilde{z}$")
    ax[2, 1].set_ylabel(L"$\tilde{z}$")

    ax[2, 1].set_xlabel(L"cross-ridge flow, $\tilde{u}$ ($\times10^{-1}$)")
    ax[2, 2].set_xlabel(L"along-ridge flow, $\tilde{v}$")
    ax[2, 3].set_xlabel(L"stratification, $\tilde{B}_\tilde{z}$ ($\times10^2$)")

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # zoomed z
    ax[1, 1].set_ylim([0, 10])
    ax[2, 1].set_ylim([0, 10])

    # fixed x
    ax[1, 1].set_xlim([-2, 2.5])
    ax[2, 1].set_xlim([-2, 2.5])
    ax[1, 2].set_xlim([-1.05, 0.05])
    ax[2, 2].set_xlim([-1.05, 0.05])
    ax[1, 3].set_xlim([-2.2, 0.5])
    ax[2, 3].set_xlim([-2.2, 0.5])

    # plot data from folder
    cases = ["can", "tc"]
    for j=1:2
        case = cases[j]
        for i=0:5
            # load
            c = loadCheckpoint1DTCNondim(string(folder, "/", case, "/checkpoint", i, ".h5"))
            τ_A = 1/c.S

            # stratification
            Bz̃ = 1 .+ differentiate(c.b̃, c.z̃)

            # colors and labels
            if i == 0
                color = "tab:red"
                label = L"$\tilde{t} = 0$"
                ls = "-"
            else
                color = colors[i, :]
                label = string(L"$\tilde{t}/\tilde{\tau}_A$ = ", Int64(round(c.t̃/τ_A)))
                ls = "-"
            end

            # plot
            ax[j, 1].plot(1e1*c.ũ,  c.z̃, c=color, ls=ls, label=label)
            ax[j, 2].plot(c.ṽ,      c.z̃, c=color, ls=ls, label=label)
            if case=="tc"
                ax[j, 2].axvline(c.Px, lw=1.0, c=color, ls="--")
            end
            ax[j, 3].plot(1e-2*Bz̃,  c.z̃, c=color, ls=ls, label=label)
        end
    end

    ax[2, 3].legend(loc=(0.05, 0.05))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    ax[2, 2].annotate(L"$P_x$", xy=(0.08, 0.1), xytext=(0.25, 0.08), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    #= ax[2, 2].annotate(L"$P_x$", xy=(0.48, 0.1), xytext=(0.2, 0.08), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->")) =#

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)
    savefig("spindownProfiles.pdf")
    println("spindownProfiles.pdf")
    close()
end

function spindownGrid(folder)
    # read data
    file = h5open(string(folder, "vs.h5"), "r")
    vs = read(file, "vs")
    ṽ_0 = read(file, "ṽ_0")
    τ_Ss = read(file, "τ_Ss")
    τ_As = read(file, "τ_As")
    close(file)

    # text outline
    outline = [pe.withStroke(linewidth=0.6, foreground="k")]

    # plot grid
    fig, ax = subplots(1, figsize=(3.404, 3.404))
    ax.set_box_aspect(1)
    ax.set_xlabel(L"spindown time, $\tilde{\tau}_S$")
    ax.set_ylabel(L"arrest time, $\tilde{\tau}_A$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    img = ax.pcolormesh(τ_Ss, τ_As, vs'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
    cb = colorbar(img, ax=ax, shrink=0.63, label=string("far-field along-slope flow,\n", L"$\tilde{v}/\tilde{v}_0$ at $\tilde{t} = 5\tilde{\tau}_A$"))
    ax.loglog([0, 1], [0, 1], transform=ax.transAxes, "w--", lw=0.5)
    ax.annotate(L"$\tilde{\tau}_A/\tilde{\tau}_S = 1$", xy=(0.9, 0.9), xytext=(0.5, 0.9), 
                xycoords="axes fraction", c="w", path_effects=outline, arrowprops=Dict("arrowstyle" => "->", "color" => "w"))
    ax.scatter(1e2, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax.scatter(5e3, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax.annotate("Fig. 6", xy=(0.3, 0.25), xycoords="axes fraction", c="w", path_effects=outline)
    ax.annotate("Fig. 7", xy=(0.8, 0.25), xycoords="axes fraction", c="w", path_effects=outline)

    tight_layout()
    #= subplots_adjust(bottom=0.2, top=0.9, left=0.0, right=0.9, wspace=0.0, hspace=0.0) =#
    savefig("spindownGrid.pdf")
    println("spindownGrid.pdf")
end

function asymmetricRidge(folder)
    # load
    c = loadCheckpoint2DPG(string(folder, "checkpoint1000.h5"))
    u, v, w = transformFromTF(c.uξ, c.uη, c.uσ)

    # plot
    ax = ridgePlot(c.chi, c.b, "", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; x=c.x, z=c.z)
    savefig("spinupRidgeAsym.pdf")
    println("spinupRidgeAsym.pdf")
    close()
end

path = "/home/hpeter/ResearchCallies/sims/" 
#= sketchRidge() =#
#= sketchSlope() =#
#= chiForSketch(string(path, "sim023/")) =#
#= chi_v_ridge(string(path, "sim021/")) =#
#= spindownProfiles(string(path, "sim024/tauA1e2_tauS5e3/")) # ratio small =#
#= spindownProfiles(string(path, "sim024/tauA1e2_tauS1e2/")) # ratio big =#
#= spindownGrid(string(path, "sim024/")) =#
#= asymmetricRidge(string(path, "sim020/")) =#
