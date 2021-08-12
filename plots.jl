using PyPlot, PyCall, Printf, HDF5, Dierckx

plt.style.use("plots.mplstyle")
close("all")
pygui(false)

include("myJuliaLib.jl")

# for loading data
include("1dtc/utils.jl")
include("1dtc_pg/utils.jl")
include("1dtc_nondim/utils.jl")
include("2dpg/setup.jl")
include("2dpg/utils.jl")
include("rayleigh/2dpg/utils.jl")
include("rayleigh/1dtc_pg/utils.jl")

# for ridgePlot
include("2dpg/plotting.jl")

# matplotlib
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

################################################################################
# Transport Constraint Paper
################################################################################

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
    v = c.uη
    ix = argmin(abs.(c.x[:, 1] .- c.L/4))

    # plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)
    ridgePlot(c.χ, c.b, "", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1], x=c.x, z=c.z, N=c.N)
    ridgePlot(v, c.b, "", L"along-ridge flow, $v$ (m s$^{-1}$)"; ax=ax[2], x=c.x, z=c.z, N=c.N)
    ax[1].plot([c.L/1e3/4, c.L/1e3/4], [c.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([c.L/1e3/4, c.L/1e3/4], [c.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")
    ax[1].set_xlim([0, c.L/1e3])
    ax[2].set_xlim([0, c.L/1e3])
    savefig("spinupRidge.pdf")
    println("spinupRidge.pdf")
    close()
end

function spinupRidgeAsym(folder)
    # load
    c = loadCheckpoint2DPG(string(folder, "checkpoint1.h5"))

    # plot
    ax = ridgePlot(c.χ, c.b, "", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; x=c.x, z=c.z, N=c.N)
    savefig("spinupRidgeAsym.pdf")
    println("spinupRidgeAsym.pdf")
    close()

    println(@sprintf("U = %1.2e m2 s-1", c.χ[1, end]))
end

function spinupProfiles(folder; σ=1)
    ii = 1:5

    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    fig.text(0.05, 0.98, string("Canonical 1D (Pr = ", σ, "):"), ha="left", va="top")
    fig.text(0.05, 0.52, string("Transport-Constrained 1D (Pr = ", σ, "):"), ha="left", va="top")

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 1].set_xlabel(string(L"streamfunction, $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow, $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification, $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    if σ == 1
        ax[1, 1].set_xlim([-5, 57])
        ax[2, 1].set_xlim([-0.1, 1.65])
        ax[1, 2].set_xlim([-2.7, 1.4])
        ax[2, 2].set_xlim([-2.7, 1.4])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    elseif σ == 200
        ax[1, 1].set_xlim([-10, 190])
        ax[2, 1].set_xlim([-5, 95])
        ax[1, 2].set_xlim([-2.0, 0.3])
        ax[2, 2].set_xlim([-2.0, 0.3])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    end

    # plot data from folder
    for i=ii
        # canonical 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/can/Pr", σ, "/checkpoint", i, ".h5"))
        label = string(Int64(c.t/86400/360), " years")
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :],     label=label)
        ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        
        # 2D PG solution
        c = loadCheckpoint2DPG(string(folder, "2dpg/Pr", σ, "/checkpoint", i, ".h5"))
        ix = argmin(abs.(c.x[:, 1] .- c.L/4))
        v = c.uη
        Bz = c.N^2 .+ differentiate(c.b[ix, :], c.z[ix, :])
        ax[1, 1].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, "k:")
        ax[1, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, "k:")
        ax[1, 3].plot(1e6*Bz,         c.z[ix, :]/1e3, "k:")

        # transport-constrained 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/tc/Pr", σ, "/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[2, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        c = loadCheckpoint2DPG(string(folder, "2dpg/Pr", σ, "/checkpoint", i, ".h5"))
        ix = argmin(abs.(c.x[:, 1] .- c.L/4))
        v = c.uη
        Bz = c.N^2 .+ differentiate(c.b[ix, :], c.z[ix, :])
        ax[2, 1].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, "k:")
        ax[2, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, "k:")
        ax[2, 3].plot(1e6*Bz,         c.z[ix, :]/1e3, "k:")
    end

    # steady state canonical
    c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/can/Pr", σ, "/checkpoint999.h5"))
    Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
    ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c="k")
    ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c="k")
    ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c="k")

    ax[2, 3].legend(loc=(0.05, 0.3))
    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["steady state", L"2D $\nu$PGCM"]
    ax[1, 3].legend(custom_handles, custom_labels, loc=(0.05, 0.72))

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

function spinupProfilesRayleigh(folder)
    ii = 1:5

    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    fig.text(0.05, 0.98, "Canonical 1D:", ha="left", va="top")
    fig.text(0.05, 0.52, "Transport-constrained 1D:", ha="left", va="top")

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 1].set_xlabel(string(L"streamfunction, $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow, $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification, $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    axins12 = inset_locator.inset_axes(ax[1, 2], width="40%", height="40%")
    axins22 = inset_locator.inset_axes(ax[2, 2], width="40%", height="40%")

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    ax[1, 1].set_xlim([0, 8.1])
    ax[2, 1].set_xlim([0, 8.1])
    ax[1, 2].set_xlim([-0.01, 0.24])
    ax[2, 2].set_xlim([-0.01, 0.24])
    ax[1, 3].set_xlim([0, 1.05])
    ax[2, 3].set_xlim([0, 1.05])
    axins12.set_xlim([-0.01, 0.005])
    axins22.set_xlim([-0.01, 0.005])
    # ax[1, 1].set_xlim([-0.7, 5])
    # ax[2, 1].set_xlim([-0.7, 5])
    # ax[1, 2].set_xlim([-0.03, 0.2])
    # ax[2, 2].set_xlim([-0.03, 0.2])
    # ax[1, 3].set_xlim([0, 1.08])
    # ax[2, 3].set_xlim([0, 1.08])
    # axins12.set_xlim([-0.022, 0.005])
    # axins22.set_xlim([-0.022, 0.005])

    # plot data from folder
    for i=ii
        # canonical 1D solution
        c = loadCheckpoint1DTCPGRayleigh(string(folder, "1dcan_pg/checkpoint", i, ".h5"))
        label = string(Int64(c.t/86400/360), " years")
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :],     label=label)
        ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        axins12.plot( 1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        c = loadCheckpoint2DPGRayleigh(string(folder, "2dpg/checkpoint", i, ".h5"))
        ix = argmin(abs.(c.x[:, 1] .- c.L/4))
        v = c.uη
        Bz = c.N^2 .+ differentiate(c.b[ix, :], c.z[ix, :])
        ax[1, 1].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, c="k", ls=":")
        ax[1, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
        axins12.plot( 1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
        ax[1, 3].plot(1e6*Bz,         c.z[ix, :]/1e3, c="k", ls=":")

        # transport-constrained 1D solution
        c = loadCheckpoint1DTCPGRayleigh(string(folder, "1dtc_pg/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[2, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        axins22.plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        c = loadCheckpoint2DPGRayleigh(string(folder, "2dpg/checkpoint", i, ".h5"))
        ix = argmin(abs.(c.x[:, 1] .- c.L/4))
        v = c.uη
        Bz = c.N^2 .+ differentiate(c.b[ix, :], c.z[ix, :])
        ax[2, 1].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, c="k", ls=":")
        ax[2, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
        axins22.plot( 1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
        ax[2, 3].plot(1e6*Bz,         c.z[ix, :]/1e3, c="k", ls=":")
    end

    # steady state canonical
    c = loadCheckpoint1DTCPGRayleigh(string(folder, "1dcan_pg/checkpoint999.h5"))
    Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
    ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")
    ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")
    ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")

    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["steady state", "2D PG"]
    ax[1, 3].legend(custom_handles, custom_labels, loc=(0.1, 0.6))
    ax[2, 3].legend(loc=(0.12, 0.3))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)
    savefig(string("spinupProfilesRayleigh.pdf"))
    println(string("spinupProfilesRayleigh.pdf"))
    close()
end

function spindownProfiles(folder; ratio=nothing)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    c = loadCheckpoint1DTCNondim(string(folder, "/tc/checkpoint1.h5"))
    fig.text(0.05, 0.98, string(L"Canonical 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top")
    fig.text(0.05, 0.52, string(L"Transport-Constrained 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top")

    ax[1, 1].set_ylabel(L"$\tilde{z}$")
    ax[2, 1].set_ylabel(L"$\tilde{z}$")

    ax[2, 1].set_xlabel(L"cross-ridge flow, $\tilde{u}$ ($\times10^{-1}$)")
    ax[2, 2].set_xlabel(L"along-ridge flow, $\tilde{v}$")
    ax[2, 3].set_xlabel(L"stratification, $\partial_{\tilde z} \tilde b$ ($\times10$)")

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # zoomed z
    ax[1, 1].set_ylim([0, 10])
    ax[2, 1].set_ylim([0, 10])

    # fixed x
    ax[1, 1].set_xlim([-0.5, 3])
    ax[2, 1].set_xlim([-0.5, 3])
    ax[1, 2].set_xlim([-1.05, 0.05])
    ax[2, 2].set_xlim([-1.05, 0.05])
    if ratio == "Small"
        ax[1, 3].set_xlim([-0.5, 0.1])
        ax[2, 3].set_xlim([-0.5, 0.1])
    elseif ratio == "Big"
        ax[1, 3].set_xlim([-25, 4])
        ax[2, 3].set_xlim([-25, 4])
    end

    # plot data from folder
    cases = ["can", "tc"]
    for j=1:2
        case = cases[j]
        for i=0:5
            # load
            c = loadCheckpoint1DTCNondim(string(folder, "/", case, "/checkpoint", i, ".h5"))
            τ_A = 1/c.S

            # stratification
            bz̃ = differentiate(c.b̃, c.z̃)

            # colors and labels
            if i == 0
                color = "tab:red"
                label = L"$\tilde{t}/\tilde\tau_A$ = 0"
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
                ax[j, 2].axvline(c.P̃x̃, lw=1.0, c=color, ls="--")
            end
            ax[j, 3].plot(1e-1*bz̃,  c.z̃, c=color, ls=ls, label=label)
        end
    end

    ax[2, 3].legend(loc=(0.05, 0.05))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    if ratio == "Small"
        ax[2, 2].annotate(L"$\partial_{\tilde x} \tilde P$", xy=(0.08, 0.8), xytext=(0.25, 0.78), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    elseif ratio == "Big"
        ax[2, 2].annotate(L"$\partial_{\tilde x} \tilde P$", xy=(0.48, 0.1), xytext=(0.2, 0.08), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    end

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)
    if ratio !== nothing
        savefig(string("spindownProfilesRatio", ratio, ".pdf"))
        println(string("spindownProfilesRatio", ratio, ".pdf"))
    else
        savefig("spindownProfiles.pdf")
        println("spindownProfiles.pdf")
    end
    close()
end

function spindownGrid(folder)
    # read data
    file = h5open(string(folder, "vs.h5"), "r")
    vs_5A = read(file, "vs_5A")
    vs_5S = read(file, "vs_5S")
    ṽ_0 = read(file, "ṽ_0")
    τ_Ss = read(file, "τ_Ss")
    τ_As = read(file, "τ_As")
    close(file)

    # text outline
    outline = [pe.withStroke(linewidth=0.6, foreground="k")]

    # aspect ratio
    aspect = log(τ_As[end]/τ_As[1])/log(τ_Ss[end]/τ_Ss[1])

    # plot grid
    fig, ax = subplots(1, 2, figsize=(3.404, 3), sharey=true)
    ax[1].set_box_aspect(aspect)
    ax[1].set_xlabel(L"spin-down time, $\tilde{\tau}_S$")
    ax[1].set_ylabel(L"arrest time, $\tilde{\tau}_A$")
    ax[1].spines["left"].set_visible(false)
    ax[1].spines["bottom"].set_visible(false)
    ax[1].set_xlim([τ_Ss[1], τ_Ss[end]])
    ax[1].set_ylim([τ_As[1], τ_As[end]])
    img = ax[1].pcolormesh(τ_Ss, τ_As, vs_5A'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
    cb = colorbar(img, ax=ax[:], shrink=0.63, label=L"far-field along-slope flow, $\tilde{v}/\tilde{v}_0$", orientation="horizontal")
    ax[1].loglog([1e1, 1e4], [1e1, 1e4], "w--", lw=0.5)
    ax[1].annotate(L"$\tilde{\tau}_A/\tilde{\tau}_S = 1$", xy=(0.7, 0.8), xytext=(0.05, 0.9), 
                xycoords="axes fraction", c="w", path_effects=outline, arrowprops=Dict("arrowstyle" => "->", "color" => "w"))
    ax[1].scatter(1e2, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[1].scatter(1e2, 2e0, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[1].annotate("Fig. 6", xy=(1.5e2, 1.9e0), xycoords="data", c="w", path_effects=outline)
    ax[1].annotate("Fig. 7", xy=(1.5e2, 0.9e2), xycoords="data", c="w", path_effects=outline)

    ax[2].set_box_aspect(aspect)
    ax[2].set_xlabel(L"spin-down time, $\tilde{\tau}_S$")
    ax[2].spines["left"].set_visible(false)
    ax[2].spines["bottom"].set_visible(false)
    ax[2].set_xlim([τ_Ss[1], τ_Ss[end]])
    ax[2].set_ylim([τ_As[1], τ_As[end]])
    img = ax[2].pcolormesh(τ_Ss, τ_As, vs_5S'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
    ax[2].loglog([1e1, 1e4], [1e1, 1e4], "w--", lw=0.5)
    ax[2].scatter(1e2, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[2].scatter(1e2, 2e0, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)

    ax[1].annotate(L"(a) $\tilde t = 5\tilde\tau_A$", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate(L"(b) $\tilde t = 5\tilde\tau_S$", (-0.04, 1.05), xycoords="axes fraction")
    
    subplots_adjust(left=0.15, right=0.85, bottom=0.35, top=0.9, wspace=0.2, hspace=0.6)

    savefig("spindownGrid.pdf")
    println("spindownGrid.pdf")
end

function spinupProfilesPGvsFull(folder)
    tDays = 1000:1000:5000
    
    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 1.9), sharey=true)

    axins1 = inset_locator.inset_axes(ax[1], width="60%", height="60%")

    ax[1].set_ylabel(L"$z$ (km)")

    ax[1].set_xlabel(string(L"cross-slope flow, $u$", "\n", L"($\times10^{-4}$ m s$^{-1}$)"))
    ax[2].set_xlabel(string(L"along-ridge flow, $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"stratification, $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(tDays, 1)))

    # fixed x
    ax[1].set_xlim([-0.2, 2])
    axins1.set_xlim([-0.2, 2])
    ax[2].set_xlim([-2.7, 2.7])
    ax[3].set_xlim([0, 1.3])

    # fixed y
    axins1.set_ylim([-1, -0.9])

    # plot data from folder
    for i=1:size(tDays, 1)
        tDay = tDays[i]
        label = string(Int64(tDay), " days")

        # 1D Full
        c = loadCheckpoint1DTC(string(folder, "1dtc/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        u = c.û*cos(c.θ)
        ax[1].plot(1e4*u,   c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        axins1.plot(1e4*u,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

        # 1D PG
        c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        u = c.û*cos(c.θ)
        ax[1].plot(1e4*u,   c.ẑ*cos(c.θ)/1e3, "k:")
        axins1.plot(1e4*u,  c.ẑ*cos(c.θ)/1e3, "k:")
        ax[2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, "k:")
        ax[3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, "k:")
    end

    ax[2].legend(loc="upper right")
    custom_labels = ["Full", "PG"]
    custom_handles = [lines.Line2D([0], [0], lw=1, ls="-", c="k"),
                      lines.Line2D([0], [0], lw=1, ls=":", c="k")]
    ax[3].legend(custom_handles, custom_labels, loc="upper left")

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.28, top=0.9, wspace=0.1, hspace=0.6)
    savefig("spinupProfilesPGvsFull.pdf")
    println("spinupProfilesPGvsFull.pdf")
    close()
end

################################################################################
# BL Theory Paper
################################################################################

function spinupRidge(folder)
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62), sharey=true)

    # L = 2000 km
    m = loadSetup2DPG(string(folder, "L2000km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L2000km/full2D/state1.h5"))
    fig.text(0.05, 0.98, string(L"$L = $", Int64(m.L/1e3), " km:"), ha="left", va="top")
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1, 1])
    ridgePlot(m, s, s.uη, "", L"along-ridge flow $v$ (m s$^{-1}$)"; ax=ax[1, 2])
    ax[1, 1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1, 2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1, 1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[1, 2].set_ylabel("")

    # L = 100 km
    m = loadSetup2DPG(string(folder, "L100km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L100km/full2D/state1.h5"))
    fig.text(0.05, 0.52, string(L"$L = $", Int64(m.L/1e3), " km:"), ha="left", va="top")
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[2, 1])
    ridgePlot(m, s, s.uη, "", L"along-ridge flow $v$ (m s$^{-1}$)"; ax=ax[2, 2])
    ax[2, 1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2, 2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2, 1].annotate("(c)", (0.0, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(d)", (0.0, 1.05), xycoords="axes fraction")
    ax[2, 2].set_ylabel("")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)

    savefig("spinupRidge.pdf")
    println("spinupRidge.pdf")
    close()
end

function spinupProfilesFull2DvsBL1D(datafilesFull2D, datafilesBL1D)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4))

    ax[1, 1].set_xlabel(L"$b$ (m s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$z$ (km)")

    ax[1, 2].set_xlabel(L"$\partial_{\hat z} B$ (s$^{-2}$)")

    ax[1, 3].set_xlabel(L"$\chi$ (m$^2$ s$^{-1}$)")

    ax[2, 1].set_xlabel(L"BL $b$ (m s$^{-2}$)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 2].set_xlabel(L"BL $\partial_{\hat z} B$ (s$^{-2}$)")

    ax[2, 3].set_xlabel(L"BL $\chi$ (m$^2$ s$^{-1}$)")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)

    c = loadCheckpoint1DTCPG(datafilesBL1D[1])
    ax[1, 1].annotate(@sprintf("Pr = %1.2e", c.Pr),                              (0.5, 0.6), xycoords="axes fraction")
    ax[1, 1].annotate(string(L"S =", @sprintf("%1.2e", c.N^2*tan(c.θ)^2/c.f^2)), (0.5, 0.5), xycoords="axes fraction")

    c = loadCheckpoint2DPG(datafilesFull2D[1])
    iξ = argmin(abs.(c.x[:, 1] .- c.L/4))

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end
    for col=2:3
        ax[1, col].set_yticklabels([])
        ax[2, col].set_yticklabels([])
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafilesFull2D, 1)-1))

    # limits
    ax[2, 1].set_ylim([0, 0.1])
    ax[2, 2].set_ylim([0, 0.1])
    ax[2, 3].set_ylim([0, 0.1])
    ax[2, 2].set_xlim([-1e-7, c.N^2/1.5])

    # plot data
    for i=1:size(datafilesFull2D, 1)
        # load
        c = loadCheckpoint2DPG(datafilesFull2D[i])
        cBL = loadCheckpoint1DTCPG(datafilesBL1D[i])

        # compute full BL solution
        S = cBL.N^2*tan(cBL.θ)^2/cBL.f^2
        bI = cBL.b
        bB = get_bB(bI, cBL.ẑ, cBL.f, cBL.θ, cBL.Pr, S, cBL.Pr*cBL.κ)
        bBL = bI + bB
        χI = -differentiate(bI, cBL.ẑ)*sin(cBL.θ)*cBL.Pr.*cBL.κ/(cBL.f^2*cos(cBL.θ)^2)
        χB = cBL.κ[1]/cBL.N^2/sin(cBL.θ)*differentiate(bB, cBL.ẑ)
        χBL = χI + χB

        # stratification
        Bz = c.N^2 .+ differentiate(c.b[iξ, :], c.z[iξ, :])
        BzBL = cBL.N^2*cos(cBL.θ) .+ differentiate(bBL, cBL.ẑ*cos(cBL.θ))

        # colors and labels
        label = string(Int64(round(c.t/86400/360)), " years")
        if i==1
            color = "k"
        else
            color = colors[i-1, :]
        end

        # plot
        ax[1, 1].plot(bBL,   (cBL.ẑ .- cBL.ẑ[1])*cos(cBL.θ)/1e3, c=color, label=label)
        ax[2, 1].plot(bBL,   (cBL.ẑ .- cBL.ẑ[1])*cos(cBL.θ)/1e3, c=color, label=label)
        ax[1, 2].plot(BzBL,  (cBL.ẑ .- cBL.ẑ[1])*cos(cBL.θ)/1e3, c=color, label=label)
        ax[2, 2].plot(BzBL,  (cBL.ẑ .- cBL.ẑ[1])*cos(cBL.θ)/1e3, c=color, label=label)
        ax[1, 3].plot(χBL,   (cBL.ẑ .- cBL.ẑ[1])*cos(cBL.θ)/1e3, c=color, label=label)
        ax[2, 3].plot(χBL,   (cBL.ẑ .- cBL.ẑ[1])*cos(cBL.θ)/1e3, c=color, label=label)
        ax[1, 1].plot(c.b[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[2, 1].plot(c.b[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[1, 2].plot(Bz,           (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[2, 2].plot(Bz,           (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[1, 3].plot(c.χ[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[2, 3].plot(c.χ[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = [L"2D $\nu$PGCM"]
    ax[1, 2].legend(custom_handles, custom_labels)
    ax[1, 3].legend()
    
    savefig("profilesSpinUpFull2DvsBL1D.pdf")
    println("profilesSpinUpFull2DvsBL1D.pdf")
end
function get_bB(bI, ẑ, f, θ, Pr, S, ν)
    z = ẑ .- ẑ[1]
    q = (f^2*cos(θ)^2*(1 + Pr*S)/4/ν[1]^2)^(1/4)
    bIz0 = differentiate_pointwise(bI[1:3], z[1:3], z[1], 1)
    bIzz0 = differentiate_pointwise(bI[1:5], z[1:5], z[1], 2)
    B = -Pr*S*bIzz0/(2*q^2)
    A = -Pr*S*bIz0/q + B
    # # approximation:
    # B = 0
    # A = -Pr*S*bIz0/q
    return @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
end

function spinupProfilesFull2DvsBL2D(datafilesFull2D, datafilesBL2D)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4))

    ax[1, 1].set_xlabel(L"$b$ (m s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$z$ (km)")

    ax[1, 2].set_xlabel(L"$\partial_{\hat z} B$ (s$^{-2}$)")

    ax[1, 3].set_xlabel(L"$\chi$ (m$^2$ s$^{-1}$)")

    ax[2, 1].set_xlabel(L"BL $b$ (m s$^{-2}$)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 2].set_xlabel(L"BL $\partial_{\hat z} B$ (s$^{-2}$)")

    ax[2, 3].set_xlabel(L"BL $\chi$ (m$^2$ s$^{-1}$)")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)

    c = loadCheckpoint2DPG(datafilesFull2D[1])
    iξ = argmin(abs.(c.x[:, 1] .- c.L/4))
    tanθ = 2*pi*0.4*c.H0/c.L
    ax[1, 1].annotate(@sprintf("Pr = %1.2e", c.Pr),                          (0.5, 0.6), xycoords="axes fraction")
    ax[1, 1].annotate(string(L"S =", @sprintf("%1.2e", c.N^2*tanθ^2/c.f^2)), (0.5, 0.5), xycoords="axes fraction")

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end
    for col=2:3
        ax[1, col].set_yticklabels([])
        ax[2, col].set_yticklabels([])
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafilesFull2D, 1)-1))

    # limits
    ax[2, 1].set_ylim([0, 0.1])
    ax[2, 2].set_ylim([0, 0.1])
    ax[2, 3].set_ylim([0, 0.1])
    ax[2, 2].set_xlim([0, 4e-7])

    # plot data
    for i=1:size(datafilesFull2D, 1)
        # load
        c = loadCheckpoint2DPG(datafilesFull2D[i])
        cBL = loadCheckpoint2DPG(datafilesBL2D[i])

        # todo: compute full BL solution

        # stratification
        Bz = c.N^2 .+ differentiate(c.b[iξ, :], c.z[iξ, :])
        BzBL = cBL.N^2 .+ differentiate(cBL.b[iξ, :], cBL.z[iξ, :])

        # colors and labels
        label = string(Int64(round(c.t/86400/360)), " years")
        if i==1
            color = "k"
        else
            color = colors[i-1, :]
        end

        # plot
        ax[1, 1].plot(cBL.b[iξ, :],   (cBL.z[iξ, :] .- cBL.z[iξ, 1])/1e3, c=color, label=label)
        ax[2, 1].plot(cBL.b[iξ, :],   (cBL.z[iξ, :] .- cBL.z[iξ, 1])/1e3, c=color, label=label)
        ax[1, 2].plot(BzBL,           (cBL.z[iξ, :] .- cBL.z[iξ, 1])/1e3, c=color, label=label)
        ax[2, 2].plot(BzBL,           (cBL.z[iξ, :] .- cBL.z[iξ, 1])/1e3, c=color, label=label)
        ax[1, 3].plot(cBL.χ[iξ, :],   (cBL.z[iξ, :] .- cBL.z[iξ, 1])/1e3, c=color, label=label)
        ax[2, 3].plot(cBL.χ[iξ, :],   (cBL.z[iξ, :] .- cBL.z[iξ, 1])/1e3, c=color, label=label)
        ax[1, 1].plot(c.b[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[2, 1].plot(c.b[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[1, 2].plot(Bz,           (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[2, 2].plot(Bz,           (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[1, 3].plot(c.χ[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
        ax[2, 3].plot(c.χ[iξ, :],   (c.z[iξ, :] .- c.z[iξ, 1])/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = [L"2D $\nu$PGCM"]
    ax[1, 2].legend(custom_handles, custom_labels)
    ax[1, 3].legend()
    
    savefig("profilesSpinUpFull2DvsBL2D.pdf")
    println("profilesSpinUpFull2DvsBL2D.pdf")
end

################################################################################
# Proposal
################################################################################

function RayleighVsFickian(datafileR, datafileF)
    # setup plot
    fig, ax = subplots(1, 2, figsize=(6.5, 2.2))

    ax[1].set_xlabel(L"$x$ (km)")
    ax[1].set_ylabel(L"$z$ (km)")
    ax[1].set_ylim([0, 2.5])

    ax[2].set_xlabel(L"along-slope flow $u^y$ ($\times 10^{-2}$ m s$^{-1}$)")
    ax[2].set_ylabel(L"$z$ (km)")
    ax[2].set_xlim([-2.1, 0.8])

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")

    # load data
    cR = loadCheckpoint1DTCPGRayleigh(datafileR)
    cF = loadCheckpoint1DTCPG(datafileF)

    # interpolate buoyancy
    θ = cR.θ
    if cF.θ != θ
        error("These simulations do not have the same slope.")
    end

    # cross-slope distance
    nx = 2^10
    nzR = size(cR.ẑ, 1)
    nzF = size(cF.ẑ, 1)
    L = 9e5
    x = 0:L/(nx - 1):L
    xxR = repeat(x, 1, nzR)
    xxF = repeat(x, 1, nzF)

    # total buoyancy arrays
    zR = repeat(cR.ẑ'*cos(θ) .+ cR.H, nx, 1) + repeat(x*tan(θ), 1, nzR)
    BR = cR.N^2*zR + repeat(cR.b', nx, 1)
    zF = repeat(cF.ẑ'*cos(θ) .+ cF.H, nx, 1) + repeat(x*tan(θ), 1, nzF)
    BF = cF.N^2*zF + repeat(cF.b', nx, 1)

    # contour plot
    levels = cR.N^2*[1000, 1500, 2000]
    ax[1].plot([0, 400], [1.0, 1.0], "k--", lw=0.5, zorder=1)
    ax[1].plot([0, 600], [1.5, 1.5], "k--", lw=0.5, zorder=1)
    ax[1].plot([0, 800], [2.0, 2.0], "k--", lw=0.5, zorder=1)
    ax[1].plot(x/1e3, x*tan(θ)/1e3,  "k-",  lw=0.5)
    ax[1].contour(xxR/1e3, zR/1e3, BR, colors="tab:blue",   levels=levels)
    ax[1].contour(xxF/1e3, zF/1e3, BF, colors="tab:orange", levels=levels)
    custom_handles = [lines.Line2D([0], [0], lw=1, ls="-", c="tab:blue"),
                      lines.Line2D([0], [0], lw=1, ls="-", c="tab:orange")]
    custom_labels = ["Rayleigh drag", "Fickian friction"]
    ax[1].legend(custom_handles, custom_labels, loc="lower right")
    ax[1].spines["bottom"].set_visible(false)
    ax[1].annotate("isopycnals", (0.05, 0.85), xycoords="axes fraction")

    # line plot
    ax[2].spines["left"].set_visible(false)
    ax[2].axvline(0, lw=0.5, ls="-", c="k")
    ax[2].plot(1e2*cR.v̂, (cR.ẑ*cos(cR.θ) .+ cR.H)/1e3, label="Rayleigh drag")
    ax[2].plot(1e2*cF.v̂, (cF.ẑ*cos(cF.θ) .+ cF.H)/1e3, label="Fickian friction")
    # ax[2].legend()

    tight_layout()

    savefig("RayleighVsFickian.pdf")
    println("RayleighVsFickian.pdf")
end

function TCRidge(folder)
    ii = 1:5

    # init plot
    fig = plt.figure(figsize=(6.5, 4))
    widths = [1.5, 2]
    gs0 = fig.add_gridspec(1, 2, width_ratios=widths, wspace=0.35)
    gs1 = gs0[1].subgridspec(2, 1)
    gs2 = gs0[2].subgridspec(2, 2, wspace=0.05)
    ax = Array{Any, 2}(undef, 2, 3)
    ax[1, 1] = fig.add_subplot(gs1[1])
    ax[2, 1] = fig.add_subplot(gs1[2])
    ax[1, 2] = fig.add_subplot(gs2[1])
    ax[1, 3] = fig.add_subplot(gs2[2])
    ax[2, 2] = fig.add_subplot(gs2[3])
    ax[2, 3] = fig.add_subplot(gs2[4])

    # ridge
    c = loadCheckpoint2DPG(string(folder, "2dpg/Pr1/checkpoint1.h5"))
    v = c.uη
    ix = argmin(abs.(c.x[:, 1] .- c.L/4))
    ridgePlot(c.χ, c.b, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1, 1], x=c.x, z=c.z, N=c.N)
    ridgePlot(v, c.b, "", L"along-ridge flow $v$ (m s$^{-1}$)"; ax=ax[2, 1], x=c.x, z=c.z, N=c.N)
    ax[1, 1].plot([c.L/1e3/4, c.L/1e3/4], [c.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2, 1].plot([c.L/1e3/4, c.L/1e3/4], [c.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1, 1].set_xlim([0, c.L/1e3])
    ax[2, 1].set_xlim([0, c.L/1e3])

    # profiles
    ax[1, 2].set_ylabel(L"$z$ (km)")
    ax[2, 2].set_ylabel(L"$z$ (km)")

    ax[1, 2].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[1, 3].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow $u^y$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"along-ridge flow $u^y$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))

    ax[1, 3].set_yticklabels([])
    ax[2, 3].set_yticklabels([])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    ax[1, 2].set_xlim([-5, 57])
    ax[1, 3].set_xlim([-0.1, 1.65])
    ax[2, 2].set_xlim([-2.7, 2.7])
    ax[2, 3].set_xlim([-2.7, 2.7])

    # plot data from folder
    for i=ii
        # canonical 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/can/Pr1/checkpoint", i, ".h5"))
        label = string(Int64(c.t/86400/360), " years")
        ax[1, 2].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        
        # 2D PG solution
        c = loadCheckpoint2DPG(string(folder, "2dpg/Pr1/checkpoint", i, ".h5"))
        ix = argmin(abs.(c.x[:, 1] .- c.L/4))
        v = c.uη
        ax[1, 2].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, "k:")
        ax[2, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, "k:")

        # transport-constrained 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/tc/Pr1/checkpoint", i, ".h5"))
        ax[1, 3].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
        ax[2, 3].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        c = loadCheckpoint2DPG(string(folder, "2dpg/Pr1/checkpoint", i, ".h5"))
        ix = argmin(abs.(c.x[:, 1] .- c.L/4))
        v = c.uη
        ax[1, 3].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, "k:")
        ax[2, 3].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, "k:")
    end

    # steady state canonical
    c = loadCheckpoint1DTCPG(string(folder, "1dtc_pg/can/Pr1/checkpoint999.h5"))
    ax[1, 2].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c="k")
    ax[2, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c="k")

    ax[2, 3].legend(loc=(0.4, 0.2))
    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["steady state", L"2D $\nu$PGCM"]
    ax[1, 3].legend(custom_handles, custom_labels, loc="upper right")

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b) Canonical", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c) Transport-Constrained", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e) Canonical", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f) Transport-Constrained", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)
    savefig(string("TCRidge.pdf"))
    println(string("TCRidge.pdf"))
    close()
end

function full2DvsBL1D(datafilesFull2D, datafilesBL1D)
    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 2.1), sharey=true)

    ax[1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    # ax[2].set_xlabel(string(L"along-ridge flow $u^y$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))
    ax[2].set_xlabel(string(L"along-ridge flow $v$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"stratification $\partial b/\partial z$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))
    ax[1].set_ylabel(L"$z$ (km)")

    c = loadCheckpoint2DPG(datafilesFull2D[1])
    iξ = argmin(abs.(c.x[:, 1] .- c.L/4))

    # limits
    ax[1].set_xlim([-0.1, 1.65])
    ax[2].set_xlim([-2.4, 0.5])
    ax[3].set_xlim([0, 1.3])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafilesFull2D, 1)))

    # plot data
    for i=1:size(datafilesFull2D, 1)
        # load
        c = loadCheckpoint2DPG(datafilesFull2D[i])
        cBL = loadCheckpoint1DTCPG(datafilesBL1D[i])

        # compute BL solution
        S = cBL.N^2*tan(cBL.θ)^2/cBL.f^2
        bI = cBL.b
        bB = get_bB(bI, cBL.ẑ, cBL.f, cBL.θ, cBL.Pr, S, cBL.Pr*cBL.κ)
        bBL = bI + bB
        χI = -differentiate(bI, cBL.ẑ)*sin(cBL.θ)*cBL.Pr.*cBL.κ/(cBL.f^2*cos(cBL.θ)^2)
        χB = cBL.κ[1]/cBL.N^2/sin(cBL.θ)*differentiate(bB, cBL.ẑ)
        χBL = χI + χB
        vBL = cumtrapz(cBL.f*cos(cBL.θ)*(χBL .- cBL.U)./(cBL.Pr*cBL.κ), cBL.ẑ)

        # stratification
        Bz = c.N^2 .+ differentiate(c.b[iξ, :], c.z[iξ, :])
        BzBL = cBL.N^2*cos(cBL.θ) .+ differentiate(bBL, cBL.ẑ*cos(cBL.θ))

        # colors and labels
        label = string(Int64(round(c.t/86400/360)), " years")
        color = colors[i, :]

        # plot
        ax[1].plot(1e3*χBL,     cBL.ẑ*cos(cBL.θ)/1e3, c=color, label=label)
        ax[2].plot(1e2*cBL.v̂,   cBL.ẑ*cos(cBL.θ)/1e3, c=color, label=label)
        ax[3].plot(1e6*BzBL,    cBL.ẑ*cos(cBL.θ)/1e3, c=color, label=label)
        ax[1].plot(1e3*c.χ[iξ, :],   c.z[iξ, :]/1e3, "k:")
        ax[2].plot(1e2*c.uη[iξ, :],  c.z[iξ, :]/1e3, "k:")
        ax[3].plot(1e6*Bz,           c.z[iξ, :]/1e3, "k:")
    end

    ax[1].legend()
    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["2D solution"]
    ax[3].legend(custom_handles, custom_labels)

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    
    subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    savefig("full2DvsBL1D.pdf")
    println("full2DvsBL1D.pdf")
end

path = "../sims/"

# sketchRidge() 
# sketchSlope() 
# chiForSketch(string(path, "sim023/")) 
# chi_v_ridge(string(path, "sim026/"))
# spinupProfiles(string(path, "sim026/"); σ=1)
# spinupProfiles(string(path, "sim026/"); σ=200)
# spinupProfilesRayleigh(string(path, "sim027/const/")) 
# spinupProfilesRayleigh(string(path, "sim027/bi/"))
# spindownProfiles(string(path, "sim033/tauA2e0_tauS1e2/"); ratio="Small")
# spindownProfiles(string(path, "sim033/tauA1e2_tauS1e2/"); ratio="Big")
# spindownGrid(string(path, "sim033/")) 
# spinupRidgeAsym(string(path, "sim031/")) 
# spinupProfilesPGvsFull(string(path, "sim025/"))
# compareChapman02Fig5a(string(path, "sim024/"))

# ii = 0:5
# θ = "5.5e-2"
# datafilesBL1D =   string.(path, "sim028/tht", θ, "/bl/checkpoint",   ii, ".h5")
# datafilesFull2D = string.(path, "sim029/tht", θ, "/full/checkpoint", ii, ".h5")
# spinupProfilesFull2DvsBL1D(datafilesFull2D, datafilesBL1D)
# # # θ = "2.7e-2"
# θ = "3.9e-2"
# datafilesBL2D =   string.(path, "sim029/tht", θ, "/bl/checkpoint",   ii, ".h5")
# datafilesFull2D = string.(path, "sim029/tht", θ, "/full/checkpoint", ii, ".h5")
# spinupProfilesFull2DvsBL2D(datafilesFull2D, datafilesBL2D)

spinupRidge(string(path, "sim034/"))
# RayleighVsFickian(string(path, "sim032/rayleigh/checkpoint1.h5"), string(path, "sim032/fickian/checkpoint1.h5"))
# TCRidge(string(path, "sim026/"))
# ii = 1:5
# θ = "2.5e-3"
# datafilesBL1D = string.(path, "sim028/tht", θ, "/bl/checkpoint", ii, ".h5")
# datafilesFull2D = string.(path, "sim026/2dpg/Pr1/checkpoint", ii, ".h5")
# full2DvsBL1D(datafilesFull2D, datafilesBL1D)