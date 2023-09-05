################################################################################
# This file contains all the code needed (appart from simulation setup) to 
# produce the plots in 
# 
#       Peterson & Callies (2022) Rapid spin up and spin down of flow along 
#       slopes. JPO. 52 (4), 579--596. doi: 10.1175/JPO-D-21-0173.1
#
################################################################################

using nuPGCM, PyPlot, PyCall, Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# for non-PG 1D simulations
# include("../non_pg_models/1d_nondim/utils.jl")
# include("../non_pg_models/1d/setup.jl")

# matplotlib
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")
pc = 1/6 # a pica is 1/6th of an inch

function sketch_ridge()
    fig, ax = subplots(1)
    ax.fill_between(x[:, 1]/1000, z[:, 1]/1000, minimum(z)/1000, color="k", alpha=0.3, lw=0.0)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_ylim([minimum(z)/1000, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    savefig("sketch_ridge.svg")
    println("sketch_ridge.svg")
end

function sketch_slope()
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
    savefig("sketch_slope.svg")
    println("sketch_slope.svg")
end

function spinup_ridge(folder)
    # load
    m = load_setup_2D(string(folder, "2dpg/mu1/setup.h5"))
    s = load_state_2D(string(folder, "2dpg/mu1/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))

    # plot
    fig, ax = subplots(1, 2, figsize=(33*pc, 33*pc/1.62/2), sharey=true)
    ridge_plot(m, s, 1e3*s.χ,  "", L"Streamfunction $\chi$ ($\times 10^{-3}$ m$^2$ s$^{-1}$)"; ax=ax[1], vext=1.5)
    ridge_plot(m, s, 1e2*s.uη, "", L"Along-slope flow $u^y$ ($\times 10^{-2}$ m s$^{-1}$)"; ax=ax[2], style="pcolormesh", vext=1.5)
    ax[1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    # ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    # ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")

    # savefig("spinup_ridge.pdf")
    # println("spinup_ridge.pdf")
    savefig("spinup_ridge.png")
    println("spinup_ridge.png")
    plt.close()
end

function spinup_ridge_asym(folder)
    # load
    m = load_setup_2D(string(folder, "setup.h5"))
    s = load_state_2D(string(folder, "state1.h5"))

    U = s.χ[1, end]

    # plot
    ax = ridge_plot(m, s, 1e3*s.χ, "", string(L"Streamfunction $\chi$ ($\times 10^{-3}$ m$^2$ s$^{-1}$)"); vext=6)
    ax.annotate(string(L"$U =$", @sprintf("%1.1f", 1e4*U), L"$\times 10^{-4}$ m$^2$ s$^{1}$"), (0.05, 0.9), size=8, xycoords="axes fraction")
    savefig("spinup_ridge_asym.pdf")
    println("spinup_ridge_asym.pdf")
    plt.close()
end

function spinup_profiles(folder; μ=1)
    ii = 1:5

    # init plot
    fig, ax = subplots(2, 3, figsize=(27*pc, 23*pc), sharey=true)

    fig.text(0.05, 0.97, string(L"Canonical 1D ($\mu$ = ", μ, "):"),             size=8, ha="left", va="top")
    fig.text(0.05, 0.51, string(L"Transport-Constrained 1D ($\mu$ = ", μ, "):"), size=8, ha="left", va="top")

    ax[1, 1].set_ylabel(L"Vertical coordinate $z$ (km)")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$ (km)")

    ax[2, 1].set_xlabel(string(L"Streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"Along-ridge flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"Stratification $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    if μ == 1
        ax[1, 1].set_xlim([-20, 57])
        ax[1, 1].set_xticks(-20:20:40)
        ax[2, 1].set_xlim([-0.5, 1.65])
        ax[2, 1].set_xticks(-0.5:0.5:1.5)
        ax[1, 2].set_xlim([-3.0, 1.7])
        ax[1, 2].set_xticks(-3:1.5:1.5)
        ax[2, 2].set_xlim([-3.0, 1.7])
        ax[2, 2].set_xticks(-3:1.5:1.5)
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    elseif μ == 200
        ax[1, 1].set_xlim([-50, 190])
        ax[1, 1].set_xticks(-50:50:150)
        ax[2, 1].set_xlim([-25, 95])
        ax[2, 1].set_xticks(-25:25:75)
        ax[1, 2].set_xlim([-2.0, 0.3])
        ax[2, 2].set_xlim([-2.0, 0.3])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    end

    # fixed y
    ax[1, 1].set_ylim([-2, 0])
    ax[2, 1].set_ylim([-2, 0])

    # setup file
    m2D = load_setup_2D(string(folder, "2dpg/mu", μ, "/setup.h5"))

    # plot data from folder
    for i=ii
        # canonical 1D solution
        m = load_setup_1D(string(folder, "1dtc_pg/can/mu", μ, "/setup.h5"))
        s = load_state_1D(string(folder, "1dtc_pg/can/mu", μ, "/state", i, ".h5"))
        label = string(Int64(m.Δt*s.i[1]/secs_in_year), " years")
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[1, 1].plot(1e3*s.χ, m.z/1e3, c=colors[i, :], label=label, zorder=0)
        ax[1, 2].plot(1e2*s.v, m.z/1e3, c=colors[i, :], label=label, zorder=0)
        ax[1, 3].plot(1e6*Bz,  m.z/1e3, c=colors[i, :], label=label, zorder=0)
        
        # 2D PG solution
        s2D = load_state_2D(string(folder, "2dpg/mu", μ, "/state", i, ".h5"))
        ix = argmin(abs.(m2D.x[:, 1] .- m2D.L/4))
        Bz2D = differentiate(s2D.b[ix, :], m2D.z[ix, :])
        ax[1, 1].plot(1e3*s2D.χ[ix, :],  m2D.z[ix, :]/1e3, "k--", lw=0.5, zorder=1)
        ax[1, 2].plot(1e2*s2D.uη[ix, :], m2D.z[ix, :]/1e3, "k--", lw=0.5, zorder=1)
        ax[1, 3].plot(1e6*Bz2D,          m2D.z[ix, :]/1e3, "k--", lw=0.5, zorder=1)

        # transport-constrained 1D solution
        m = load_setup_1D(string(folder, "1dtc_pg/tc/mu", μ, "/setup.h5"))
        s = load_state_1D(string(folder, "1dtc_pg/tc/mu", μ, "/state", i, ".h5"))
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[2, 1].plot(1e3*s.χ, m.z/1e3, c=colors[i, :], label=label)
        ax[2, 2].plot(1e2*s.v, m.z/1e3, c=colors[i, :], label=label)
        ax[2, 3].plot(1e6*Bz,  m.z/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        ax[2, 1].plot(1e3*s2D.χ[ix, :],  m2D.z[ix, :]/1e3, "k--", lw=0.5)
        ax[2, 2].plot(1e2*s2D.uη[ix, :], m2D.z[ix, :]/1e3, "k--", lw=0.5)
        ax[2, 3].plot(1e6*Bz2D,          m2D.z[ix, :]/1e3, "k--", lw=0.5)
    end

    # steady state canonical
    m = load_setup_1D(string(folder, "1dtc_pg/can/mu", μ, "/setup.h5"))
    s = load_state_1D(string(folder, "1dtc_pg/can/mu", μ, "/state-1.h5"))
    Bz = m.N2 .+ differentiate(s.b, m.z)
    ax[1, 1].plot(1e3*s.χ,  m.z/1e3, c="k")
    ax[1, 2].plot(1e2*s.v,  m.z/1e3, c="k")
    ax[1, 3].plot(1e6*Bz,   m.z/1e3, c="k")

    ax[2, 3].legend(loc="upper left")
    custom_handles = [lines.Line2D([0], [0], c="k", ls="--", lw=0.5)]
    custom_labels = [L"2D $\nu$PGCM"]
    ax[1, 3].legend(custom_handles, custom_labels, loc=(0.01, 0.8))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    if μ == 1
        ax[1, 3].annotate("Steady state", xy=(0.4, 0.35), xytext=(0.05, 0.6), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    elseif μ == 200
        ax[1, 3].annotate("Steady state", xy=(0.5, 0.35), xytext=(0.05, 0.6), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    end

    subplots_adjust(hspace=0.4)
    savefig(string("spinup_profilesMu", μ, ".pdf"))
    println(string("spinup_profilesMu", μ, ".pdf"))
    plt.close()
end

function spindown_profiles(folder; ratio=nothing)
    # init plot
    fig, ax = subplots(2, 3, figsize=(27*pc, 23*pc), sharey=true)

    c = loadCheckpoint1DTCNondim(string(folder, "/tc/checkpoint1.h5"))
    fig.text(0.05, 0.97, string(L"Canonical 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $ ", @sprintf("%1.2f", 1/c.H/c.S), "):"),             ha="left", va="top", size=8)
    fig.text(0.05, 0.51, string(L"Transport-Constrained 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $ ", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top", size=8)

    ax[1, 1].set_ylabel(L"Vertical coordinate $\tilde{z}$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $\tilde{z}$")

    ax[2, 1].set_xlabel(L"Cross-slope flow $\tilde{u}$ ($\times10^{-1}$)")
    ax[2, 2].set_xlabel(L"Along-slope flow $\tilde{v}$")
    ax[2, 3].set_xlabel(L"Stratification $\partial_{\tilde z} \tilde b$ ($\times10$)")

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # zoomed z
    ax[1, 1].set_ylim([0, 10])
    ax[2, 1].set_ylim([0, 10])

    # fixed x
    ax[1, 1].set_xlim([-1, 2.8])
    ax[1, 1].set_xticks(-1:1:2)
    ax[2, 1].set_xlim([-1, 2.8])
    ax[2, 1].set_xticks(-1:1:2)
    ax[1, 2].set_xlim([-1.5, 0.05])
    ax[1, 2].set_xticks(-1.5:0.5:0)
    ax[2, 2].set_xlim([-1.5, 0.05])
    ax[2, 2].set_xticks(-1.5:0.5:0)
    if ratio == "Small"
        ax[1, 3].set_xlim([-0.6, 0.1])
        ax[1, 3].set_xticks(-0.6:0.2:0.0)
        ax[2, 3].set_xlim([-0.6, 0.1])
        ax[2, 3].set_xticks(-0.6:0.2:0.0)
    elseif ratio == "Big"
        ax[1, 3].set_xlim([-30, 4])
        ax[1, 3].set_xticks(-30:10:0)
        ax[2, 3].set_xlim([-30, 4])
        ax[2, 3].set_xticks(-30:10:0)
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

    ax[1, 3].legend(loc="upper left")

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    if ratio == "Small"
        ax[2, 2].annotate(L"$\partial_{\tilde x} \tilde P$", xy=(0.38, 0.8), xytext=(0.6, 0.8), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    elseif ratio == "Big"
        ax[2, 2].annotate(L"$\partial_{\tilde x} \tilde P$", xy=(0.6, 0.1), xytext=(0.05, 0.1), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    end

    subplots_adjust(hspace=0.4)
    if ratio !== nothing
        savefig(string("spindown_profiles_ratio", ratio, ".pdf"))
        println(string("spindown_profiles_ratio", ratio, ".pdf"))
    else
        savefig("spindown_profiles.pdf")
        println("spindown_profiles.pdf")
    end
    plt.close()
end

function spindown_grid(folder)
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
    fig, ax = subplots(1, 2, figsize=(19*pc, 16*pc), sharey=true)
    ax[1].set_box_aspect(aspect)
    ax[1].set_xlabel(L"Spin-down time $\tilde{\tau}_S$")
    ax[1].set_ylabel(L"Arrest time $\tilde{\tau}_A$")
    ax[1].spines["left"].set_visible(false)
    ax[1].spines["bottom"].set_visible(false)
    ax[1].set_xlim([τ_Ss[1], τ_Ss[end]])
    ax[1].set_ylim([τ_As[1], τ_As[end]])
    img = ax[1].pcolormesh(τ_Ss, τ_As, vs_5A'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
    cb = colorbar(img, ax=ax[:], shrink=0.63, label=L"Far-field along-slope flow $\tilde{v}/\tilde{v}_0$", orientation="horizontal")
    ax[1].loglog([1e1, 1e4], [1e1, 1e4], "w--", lw=0.5)
    ax[1].annotate(L"$\tilde{\tau}_A/\tilde{\tau}_S = 1$", xy=(0.7, 0.8), xytext=(0.05, 0.9), 
                xycoords="axes fraction", c="w", path_effects=outline, arrowprops=Dict("arrowstyle" => "->", "color" => "w"))
    ax[1].scatter(1e2, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[1].scatter(1e2, 2e0, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[1].annotate("Fig. 6", xy=(1.5e2, 1.9e0), xycoords="data", c="w", path_effects=outline)
    ax[1].annotate("Fig. 7", xy=(1.5e2, 0.9e2), xycoords="data", c="w", path_effects=outline)

    ax[2].set_box_aspect(aspect)
    ax[2].set_xlabel(L"Spin-down time $\tilde{\tau}_S$")
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
    
    subplots_adjust(bottom=0.4, wspace=-0.2)

    savefig("spindown_grid.pdf")
    println("spindown_grid.pdf")
    plt.close()
end

function spinup_profiles_PGvsFull(folder)
    tDays = 1000:1000:5000
    
    # init plot
    fig, ax = subplots(1, 3, figsize=(27*pc, 14*pc), sharey=true)

    axins1 = inset_locator.inset_axes(ax[1], width="60%", height="60%")

    ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")

    ax[1].set_xlabel(string(L"Cross-ridge flow $u$", "\n", L"($\times10^{-4}$ m s$^{-1}$)"))
    ax[2].set_xlabel(string(L"Along-ridge flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"Stratification $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(tDays, 1)))

    # fixed x
    ax[1].set_xlim([-0.2, 2.3])
    axins1.set_xlim([-1.0, 2.3])
    axins1.set_xticks(-1:2)
    axins1.set_xticklabels(["−1", "0", "1", "2"], size=5)
    ax[2].set_xlim([-3, 2.9])
    ax[2].set_xticks(-3:1.5:1.5)
    ax[3].set_xlim([0, 1.3])

    # fixed y
    axins1.set_ylim([-1, -0.9])
    ax[1].set_ylim([-1, 0.0])

    # yticks
    axins1.set_yticks([-1, -0.95, -0.9])
    axins1.set_yticklabels(["−1", "−0.95", "−0.9"], size=6)

    # plot data from folder
    for i=1:size(tDays, 1)
        tDay = tDays[i]
        label = string(Int64(tDay), " days")

        # 1D Full
        m = loadSetup1D(string(folder, "1dtc/setup.h5"))
        s = loadState1D(string(folder, "1dtc/state", i, ".h5"))
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[1].plot(1e4*s.u,  m.z/1e3, c=colors[i, :], label=label)
        axins1.plot(1e4*s.u, m.z/1e3, c=colors[i, :], label=label)
        ax[2].plot(1e2*s.v,  m.z/1e3, c=colors[i, :], label=label)
        ax[3].plot(1e6*Bz,   m.z/1e3, c=colors[i, :], label=label)

        # 1D PG
        m = load_setup_1D(string(folder, "1dtc_pg/setup.h5"))
        s = load_state_1D(string(folder, "1dtc_pg/state", i, ".h5"))
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[1].plot(1e4*s.u,   m.z/1e3, "k--", lw=0.5)
        axins1.plot(1e4*s.u,  m.z/1e3, "k--", lw=0.5)
        ax[2].plot(1e2*s.v,   m.z/1e3, "k--", lw=0.5)
        ax[3].plot(1e6*Bz,    m.z/1e3, "k--", lw=0.5)
    end

    ax[2].legend(loc="upper right")
    custom_labels = ["Full", "PG"]
    custom_handles = [lines.Line2D([0], [0], lw=1, ls="-", c="k"),
                      lines.Line2D([0], [0], lw=0.5, ls="--", c="k")]
    ax[3].legend(custom_handles, custom_labels, loc="upper left")

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.28, top=0.9, wspace=0.1, hspace=0.6)
    savefig("spinup_profiles_PGvsFull.pdf")
    println("spinup_profiles_PGvsFull.pdf")
    plt.close()
end

path = "../../sims/"

# sketchRidge() 
# sketchSlope() 
spinup_ridge(string(path, "sim039/"))
# spinup_ridge_asym(string(path, "sim040/")) 
# spinup_profiles(string(path, "sim039/"); μ=1)
# spinup_profiles(string(path, "sim039/"); μ=200)
# spindown_profiles(string(path, "sim033/tauA2e0_tauS1e2/"); ratio="Small")
# spindown_profiles(string(path, "sim033/tauA1e2_tauS1e2/"); ratio="Big")
# spindown_grid(string(path, "sim033/")) 
# spinup_profiles_PGvsFull(string(path, "sim025/"))