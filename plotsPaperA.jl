using PyPlot, PyCall, Printf, HDF5, Dierckx

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

include("myJuliaLib.jl")

# for loading data
include("1dtc/utils.jl")
include("1dtc_pg/setup.jl")
include("1dtc_nondim/utils.jl")
include("2dpg/setup.jl")
include("rayleigh/2dpg/utils.jl")
include("rayleigh/1dtc_pg/utils.jl")

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

function spinupRidge(folder)
    # load
    m = loadSetup2DPG(string(folder, "const/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "const/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))

    # plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)
    ridgePlot(m, s, 1e3*s.χ,  "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"); ax=ax[1])
    ridgePlot(m, s, 1e2*s.uη, "", string(L"along-ridge flow $v$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"); ax=ax[2], style="pcolormesh")
    ax[1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")
    tight_layout()
    savefig("spinupRidge.pdf")
    println("spinupRidge.pdf")
    plt.close()
end

function spinupRidgeAsym(folder)
    # load
    m = loadSetup2DPG(string(folder, "setup.h5"))
    s = loadState2DPG(string(folder, "state1.h5"))

    # plot
    ax = ridgePlot(m, s, 1e3*s.χ, "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    savefig("spinupRidgeAsym.pdf")
    println("spinupRidgeAsym.pdf")
    plt.close()

    println(@sprintf("U = %1.2e m2 s-1", s.χ[1, end]))
end

function spinupProfiles(folder; μ=1)
    ii = 1:5

    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    fig.text(0.05, 0.98, string(L"Canonical 1D ($\mu$ = ", μ, "):"), ha="left", va="top")
    fig.text(0.05, 0.52, string(L"Transport-Constrained 1D ($\mu$ = ", μ, "):"), ha="left", va="top")

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    if μ == 1
        ax[1, 1].set_xlim([-5, 57])
        ax[2, 1].set_xlim([-0.1, 1.65])
        ax[1, 2].set_xlim([-2.7, 1.4])
        ax[2, 2].set_xlim([-2.7, 1.4])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    elseif μ == 200
        ax[1, 1].set_xlim([-10, 190])
        ax[2, 1].set_xlim([-5, 95])
        ax[1, 2].set_xlim([-2.0, 0.3])
        ax[2, 2].set_xlim([-2.0, 0.3])
        ax[1, 3].set_xlim([0, 1.3])
        ax[2, 3].set_xlim([0, 1.3])
    end

    # setup file
    m2D = loadSetup2DPG(string(folder, "2dpg/mu", μ, "/setup.h5"))

    # plot data from folder
    for i=ii
        # canonical 1D solution
        m = loadSetup1DPG(string(folder, "1dtc_pg/can/mu", μ, "/setup.h5"))
        s = loadState1DPG(string(folder, "1dtc_pg/can/mu", μ, "/state", i, ".h5"))
        label = string(Int64(m.Δt*(s.i[1] - 1)/secsInYear), " years")
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[1, 1].plot(1e3*s.χ, m.z/1e3, c=colors[i, :], label=label)
        ax[1, 2].plot(1e2*s.v, m.z/1e3, c=colors[i, :], label=label)
        ax[1, 3].plot(1e6*Bz,  m.z/1e3, c=colors[i, :], label=label)
        
        # 2D PG solution
        s2D = loadState2DPG(string(folder, "2dpg/mu", μ, "/state", i, ".h5"))
        ix = argmin(abs.(m2D.x[:, 1] .- m2D.L/4))
        Bz2D = differentiate(s2D.b[ix, :], m2D.z[ix, :])
        ax[1, 1].plot(1e3*s2D.χ[ix, :],  m2D.z[ix, :]/1e3, "k:")
        ax[1, 2].plot(1e2*s2D.uη[ix, :], m2D.z[ix, :]/1e3, "k:")
        ax[1, 3].plot(1e6*Bz2D,          m2D.z[ix, :]/1e3, "k:")

        # transport-constrained 1D solution
        m = loadSetup1DPG(string(folder, "1dtc_pg/tc/mu", μ, "/setup.h5"))
        s = loadState1DPG(string(folder, "1dtc_pg/tc/mu", μ, "/state", i, ".h5"))
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[2, 1].plot(1e3*s.χ, m.z/1e3, c=colors[i, :], label=label)
        ax[2, 2].plot(1e2*s.v, m.z/1e3, c=colors[i, :], label=label)
        ax[2, 3].plot(1e6*Bz,  m.z/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        ax[2, 1].plot(1e3*s2D.χ[ix, :],  m2D.z[ix, :]/1e3, "k:")
        ax[2, 2].plot(1e2*s2D.uη[ix, :], m2D.z[ix, :]/1e3, "k:")
        ax[2, 3].plot(1e6*Bz2D,          m2D.z[ix, :]/1e3, "k:")
    end

    # steady state canonical
    m = loadSetup1DPG(string(folder, "1dtc_pg/can/mu", μ, "/setup.h5"))
    s = loadState1DPG(string(folder, "1dtc_pg/can/mu", μ, "/state-1.h5"))
    Bz = m.N2 .+ differentiate(s.b, m.z)
    ax[1, 1].plot(1e3*s.χ,  m.z/1e3, c="k")
    ax[1, 2].plot(1e2*s.v,  m.z/1e3, c="k")
    ax[1, 3].plot(1e6*Bz,   m.z/1e3, c="k")

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
    savefig(string("spinupProfilesMu", μ, ".pdf"))
    println(string("spinupProfilesMu", μ, ".pdf"))
    plt.close()
end

# function spinupProfilesRayleigh(folder)
#     ii = 1:5

#     # init plot
#     fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

#     fig.text(0.05, 0.98, "Canonical 1D:", ha="left", va="top")
#     fig.text(0.05, 0.52, "Transport-constrained 1D:", ha="left", va="top")

#     ax[1, 1].set_ylabel(L"$z$ (km)")
#     ax[2, 1].set_ylabel(L"$z$ (km)")

#     ax[2, 1].set_xlabel(string(L"streamfunction, $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
#     ax[2, 2].set_xlabel(string(L"along-ridge flow, $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
#     ax[2, 3].set_xlabel(string(L"stratification, $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

#     axins12 = inset_locator.inset_axes(ax[1, 2], width="40%", height="40%")
#     axins22 = inset_locator.inset_axes(ax[2, 2], width="40%", height="40%")

#     # color map
#     colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

#     # fixed x
#     ax[1, 1].set_xlim([0, 8.1])
#     ax[2, 1].set_xlim([0, 8.1])
#     ax[1, 2].set_xlim([-0.01, 0.24])
#     ax[2, 2].set_xlim([-0.01, 0.24])
#     ax[1, 3].set_xlim([0, 1.05])
#     ax[2, 3].set_xlim([0, 1.05])
#     axins12.set_xlim([-0.01, 0.005])
#     axins22.set_xlim([-0.01, 0.005])
#     # ax[1, 1].set_xlim([-0.7, 5])
#     # ax[2, 1].set_xlim([-0.7, 5])
#     # ax[1, 2].set_xlim([-0.03, 0.2])
#     # ax[2, 2].set_xlim([-0.03, 0.2])
#     # ax[1, 3].set_xlim([0, 1.08])
#     # ax[2, 3].set_xlim([0, 1.08])
#     # axins12.set_xlim([-0.022, 0.005])
#     # axins22.set_xlim([-0.022, 0.005])

#     # plot data from folder
#     for i=ii
#         # canonical 1D solution
#         c = loadCheckpoint1DTCPGRayleigh(string(folder, "1dcan_pg/checkpoint", i, ".h5"))
#         label = string(Int64(c.t/86400/360), " years")
#         Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
#         ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :],     label=label)
#         ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
#         axins12.plot( 1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
#         ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

#         # 2D PG solution
#         c = loadCheckpoint2DPGRayleigh(string(folder, "2dpg/checkpoint", i, ".h5"))
#         ix = argmin(abs.(c.x[:, 1] .- c.L/4))
#         v = c.uη
#         Bz = c.N^2 .+ differentiate(c.b[ix, :], c.z[ix, :])
#         ax[1, 1].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, c="k", ls=":")
#         ax[1, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
#         axins12.plot( 1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
#         ax[1, 3].plot(1e6*Bz,         c.z[ix, :]/1e3, c="k", ls=":")

#         # transport-constrained 1D solution
#         c = loadCheckpoint1DTCPGRayleigh(string(folder, "1dtc_pg/checkpoint", i, ".h5"))
#         Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
#         ax[2, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
#         ax[2, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
#         axins22.plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)
#         ax[2, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c=colors[i, :], label=label)

#         # 2D PG solution
#         c = loadCheckpoint2DPGRayleigh(string(folder, "2dpg/checkpoint", i, ".h5"))
#         ix = argmin(abs.(c.x[:, 1] .- c.L/4))
#         v = c.uη
#         Bz = c.N^2 .+ differentiate(c.b[ix, :], c.z[ix, :])
#         ax[2, 1].plot(1e3*c.χ[ix, :], c.z[ix, :]/1e3, c="k", ls=":")
#         ax[2, 2].plot(1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
#         axins22.plot( 1e2*v[ix, :],   c.z[ix, :]/1e3, c="k", ls=":")
#         ax[2, 3].plot(1e6*Bz,         c.z[ix, :]/1e3, c="k", ls=":")
#     end

#     # steady state canonical
#     c = loadCheckpoint1DTCPGRayleigh(string(folder, "1dcan_pg/checkpoint999.h5"))
#     Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
#     ax[1, 1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")
#     ax[1, 2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")
#     ax[1, 3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, c="k", label="steady state")

#     custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
#                       lines.Line2D([0], [0], c="k", ls=":", lw="1")]
#     custom_labels = ["steady state", "2D PG"]
#     ax[1, 3].legend(custom_handles, custom_labels, loc=(0.1, 0.6))
#     ax[2, 3].legend(loc=(0.12, 0.3))

#     ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
#     ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
#     ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
#     ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
#     ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
#     ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

#     subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)
#     savefig(string("spinupProfilesRayleigh.pdf"))
#     println(string("spinupProfilesRayleigh.pdf"))
#     plt.close()
# end

function spindownProfiles(folder; ratio=nothing)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4), sharey=true)

    c = loadCheckpoint1DTCNondim(string(folder, "/tc/checkpoint1.h5"))
    fig.text(0.05, 0.98, string(L"Canonical 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top")
    fig.text(0.05, 0.52, string(L"Transport-Constrained 1D $(\tilde{\tau}_A/\tilde{\tau}_S = $", @sprintf("%1.2f", 1/c.H/c.S), "):"), ha="left", va="top")

    ax[1, 1].set_ylabel(L"$\tilde{z}$")
    ax[2, 1].set_ylabel(L"$\tilde{z}$")

    ax[2, 1].set_xlabel(L"cross-ridge flow $\tilde{u}$ ($\times10^{-1}$)")
    ax[2, 2].set_xlabel(L"along-ridge flow $\tilde{v}$")
    ax[2, 3].set_xlabel(L"stratification $\partial_{\tilde z} \tilde b$ ($\times10$)")

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
    plt.close()
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
    ax[1].set_xlabel(L"spin-down time $\tilde{\tau}_S$")
    ax[1].set_ylabel(L"arrest time $\tilde{\tau}_A$")
    ax[1].spines["left"].set_visible(false)
    ax[1].spines["bottom"].set_visible(false)
    ax[1].set_xlim([τ_Ss[1], τ_Ss[end]])
    ax[1].set_ylim([τ_As[1], τ_As[end]])
    img = ax[1].pcolormesh(τ_Ss, τ_As, vs_5A'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
    cb = colorbar(img, ax=ax[:], shrink=0.63, label=L"far-field along-slope flow $\tilde{v}/\tilde{v}_0$", orientation="horizontal")
    ax[1].loglog([1e1, 1e4], [1e1, 1e4], "w--", lw=0.5)
    ax[1].annotate(L"$\tilde{\tau}_A/\tilde{\tau}_S = 1$", xy=(0.7, 0.8), xytext=(0.05, 0.9), 
                xycoords="axes fraction", c="w", path_effects=outline, arrowprops=Dict("arrowstyle" => "->", "color" => "w"))
    ax[1].scatter(1e2, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[1].scatter(1e2, 2e0, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax[1].annotate("Fig. 6", xy=(1.5e2, 1.9e0), xycoords="data", c="w", path_effects=outline)
    ax[1].annotate("Fig. 7", xy=(1.5e2, 0.9e2), xycoords="data", c="w", path_effects=outline)

    ax[2].set_box_aspect(aspect)
    ax[2].set_xlabel(L"spin-down time $\tilde{\tau}_S$")
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

    ax[1].set_xlabel(string(L"cross-slope flow $u$", "\n", L"($\times10^{-4}$ m s$^{-1}$)"))
    ax[2].set_xlabel(string(L"along-ridge flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"stratification $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

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
        m = loadSetup1DPG(string(folder, "1dtc_pg/setup.h5"))
        s = loadState1DPG(string(folder, "1dtc_pg/state", i, ".h5"))
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[1].plot(1e4*s.u,   m.z/1e3, "k:")
        axins1.plot(1e4*s.u,  m.z/1e3, "k:")
        ax[2].plot(1e2*s.v,   m.z/1e3, "k:")
        ax[3].plot(1e6*Bz,    m.z/1e3, "k:")
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
    plt.close()
end

path = "../sims/"

# sketchRidge() 
# sketchSlope() 
# spinupRidge(string(path, "sim037/"))
# spinupProfiles(string(path, "sim039/"); μ=1)
# spinupProfiles(string(path, "sim039/"); μ=200)
# spinupProfilesRayleigh(string(path, "sim027/const/")) 
# spinupProfilesRayleigh(string(path, "sim027/bi/"))
# spindownProfiles(string(path, "sim033/tauA2e0_tauS1e2/"); ratio="Small")
# spindownProfiles(string(path, "sim033/tauA1e2_tauS1e2/"); ratio="Big")
# spindownGrid(string(path, "sim033/")) 
# spinupRidgeAsym(string(path, "sim031/")) 
spinupProfilesPGvsFull(string(path, "sim025/"))