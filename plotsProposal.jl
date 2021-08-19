using PyPlot, PyCall, Printf, HDF5, Dierckx

plt.style.use("plots.mplstyle")
plt.close()("all")
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

function boundaryCorrection(χI::Array{Float64,1}, z::Array{Float64,1}, q::Float64)
    A = -χI[1]
    χIz0 = differentiate_pointwise(χI[1:3], z[1:3], z[1], 1)
    B = -χIz0/q + A
    χB = @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
    return χB
end

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
    plt.close()()
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

# RayleighVsFickian(string(path, "sim032/rayleigh/checkpoint1.h5"), string(path, "sim032/fickian/checkpoint1.h5"))
# TCRidge(string(path, "sim026/"))
# ii = 1:5
# θ = "2.5e-3"
# datafilesBL1D = string.(path, "sim028/tht", θ, "/bl/checkpoint", ii, ".h5")
# datafilesFull2D = string.(path, "sim026/2dpg/Pr1/checkpoint", ii, ".h5")
# full2DvsBL1D(datafilesFull2D, datafilesBL1D)