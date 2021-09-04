
using PyPlot, PyCall, Printf, HDF5, Dierckx

plt.style.use("plots.mplstyle")
plt.close("all")
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

# utility function
function boundaryCorrection(χI::Array{Float64,1}, z::Array{Float64,1}, q::Float64)
    A = -χI[1]
    χIz0 = differentiate_pointwise(χI[1:3], z[1:3], z[1], 1)
    B = -χIz0/q + A
    χB = @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
    return χB
end

function ridge(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    m = loadSetup2DPG(string(folder, "L2000km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L2000km/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1])
    ridgePlot(m, s, s.uη, "", L"along-ridge flow $u^\eta$ (m s$^{-1}$)"; ax=ax[2])
    ax[1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")

    subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    savefig("ridge.pdf")
    println("ridge.pdf")
    plt.close()
end

function ridgeFull2DvsBL1D(folder)
    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 6.5/1.62/2), sharey=true)

    ax[1].set_ylabel(L"$z$ (km)")
    ax[1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2].set_xlabel(string(L"along-ridge flow $u^\eta$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # limits
    ax[1].set_xlim([-0.05, 1.7])
    ax[2].set_xlim([-2.5, 0.5])
    ax[3].set_xlim([0, 1.5])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    m = loadSetup2DPG(string(folder, "L2000km/full2D/setup.h5"))
    ix= argmin(abs.(m.x[:, 1] .- m.L/4))

    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 1D
        c = loadCheckpoint1DTCPG(string(folder, "L2000km/bl1D/checkpoint$i.h5"))
        ẑ = c.ẑ .- c.ẑ[1]
        q = (1/(4*c.Pr*c.κ[1])*(c.f^2*cos(c.θ)^2/(c.Pr*c.κ[1]) + c.N^2*sin(c.θ)^2/c.κ[1]))^(1/4)
        bI = c.b
        χI = -differentiate(bI, c.ẑ)*sin(c.θ)*c.Pr.*c.κ/(c.f^2*cos(c.θ)^2)
        χB = boundaryCorrection(χI, ẑ, q)
        bB = cumtrapz(χB*c.N^2*sin(c.θ)/c.κ[1], ẑ) .- trapz(χB*c.N^2*sin(c.θ)/c.κ[1], ẑ)
        χ = χI + χB
        b = bI + bB
        Bz = c.N^2*cos(c.θ) .+ differentiate(b, ẑ*cos(c.θ))
        v = cumtrapz(c.f*cos(c.θ)*(χ .- c.U)./(c.Pr*c.κ), ẑ)
        label = string(Int64(c.t/86400/360), " years")
        ax[1].plot(1e3*χ,   ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, c=color, label=label)
        ax[2].plot(1e2*v,   ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, c=color, label=label)
        ax[3].plot(1e6*Bz,  ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, c=color, label=label)

        # full 2D
        s = loadState2DPG(string(folder, "L2000km/full2D/state$i.h5"))
        bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax[1].plot(1e3*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        ax[2].plot(1e2*s.uη[ix, :],  m.z[ix, :]/1e3, "k:")
        ax[3].plot(1e6*bz,           m.z[ix, :]/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL 1D", "full 2D"]
    ax[2].legend(custom_handles, custom_labels)
    ax[1].legend()
    
    savefig("ridgeFull2DvsBL1D.pdf")
    println("ridgeFull2DvsBL1D.pdf")
    plt.close()
end


function seamount(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L200km/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1])
    ridgePlot(m, s, s.uη, "", L"along-ridge flow $u^\eta$ (m s$^{-1}$)"; ax=ax[2])
    ax[1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")

    subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    savefig("seamount.pdf")
    println("seamount.pdf")
    plt.close()
end

function seamountFull2DvsBL(folder)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 6.5/1.62), sharey=true)

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-2}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow $u^\eta$", "\n", L"($\times 10^{-1}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    fig.text(0.05, 0.98, "BL 1D:", ha="left", va="top")
    fig.text(0.05, 0.52, "BL 2D:", ha="left", va="top")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # limits
    ax[1, 1].set_xlim([-1.7, 0.05])
    ax[1, 2].set_xlim([0, 2.6])
    ax[1, 3].set_xlim([0, 1.2])
    ax[2, 1].set_xlim([-1.7, 0.05])
    ax[2, 2].set_xlim([0, 2.6])
    ax[2, 3].set_xlim([0, 1.2])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # model setups
    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    mBL = loadSetup2DPG(string(folder, "L200km/bl2D/setup.h5"))
    ixBL = argmin(abs.(mBL.x[:, 1] .- mBL.L/4))

    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 1D
        c = loadCheckpoint1DTCPG(string(folder, "L200km/bl1D/checkpoint$i.h5"))
        ẑ = c.ẑ .- c.ẑ[1]
        q = (1/(4*c.Pr*c.κ[1])*(c.f^2*cos(c.θ)^2/(c.Pr*c.κ[1]) + c.N^2*sin(c.θ)^2/c.κ[1]))^(1/4)
        bI = c.b
        χI = -differentiate(bI, c.ẑ)*sin(c.θ)*c.Pr.*c.κ/(c.f^2*cos(c.θ)^2)
        χB = boundaryCorrection(χI, ẑ, q)
        bB = cumtrapz(χB*c.N^2*sin(c.θ)/c.κ[1], ẑ) .- trapz(χB*c.N^2*sin(c.θ)/c.κ[1], ẑ)
        χ = χI + χB
        b = bI + bB
        Bz = c.N^2*cos(c.θ) .+ differentiate(b, ẑ*cos(c.θ))
        v = cumtrapz(c.f*cos(c.θ)*(χ .- c.U)./(c.Pr*c.κ), ẑ)
        label = string(Int64(c.t/86400/360), " years")
        ax[1, 1].plot(1e2*χ,   ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, c=color, label=label)
        ax[1, 2].plot(1e1*v,   ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, c=color, label=label)
        ax[1, 3].plot(1e6*Bz,  ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, c=color, label=label)

        # full 2D
        s = loadState2DPG(string(folder, "L200km/full2D/state$i.h5"))
        bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax[1, 1].plot(1e2*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        ax[1, 2].plot(1e1*s.uη[ix, :],  m.z[ix, :]/1e3, "k:")
        ax[1, 3].plot(1e6*bz,           m.z[ix, :]/1e3, "k:")

        # BL 2D
        s = loadState2DPG(string(folder, "L200km/bl2D/state$i.h5"))
        bI = s.b[ixBL, :]
        χI = s.χ[ixBL, :]
        bIξ = ξDerivative(mBL, s.b)
        q = (1/(4*mBL.ν[ixBL, 1])*(mBL.f^2/mBL.ν[ixBL, 1] - mBL.Hx[ixBL]*bIξ[ixBL, 1]/mBL.H[ixBL]/mBL.κ[ixBL, 1]))^(1/4)
        χB = boundaryCorrection(χI, mBL.z[ixBL, :] .- mBL.z[ixBL, 1], q)
        bB = cumtrapz(χB*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) .- trapz(χB.*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) 
        χ = χI + χB
        b = bI + bB
        bz = differentiate(b, mBL.z[ixBL, :])
        ax[2, 1].plot(1e2*χ,             mBL.z[ixBL, :]/1e3, c=color, label=label)
        ax[2, 2].plot(1e1*s.uη[ixBL, :], mBL.z[ixBL, :]/1e3, c=color, label=label)
        ax[2, 3].plot(1e6*bz,            mBL.z[ixBL, :]/1e3, c=color, label=label)

        # full 2D
        s = loadState2DPG(string(folder, "L200km/full2D/state$i.h5"))
        bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax[2, 1].plot(1e2*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        ax[2, 2].plot(1e1*s.uη[ix, :],  m.z[ix, :]/1e3, "k:")
        ax[2, 3].plot(1e6*bz,           m.z[ix, :]/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["full 2D"]
    ax[1, 2].legend(custom_handles, custom_labels)
    ax[1, 1].legend()
    
    savefig("seamountFull2DvsBL.pdf")
    println("seamountFull2DvsBL.pdf")
    plt.close()
end


path = "../sims/"

# ridge(string(path, "sim034/"))
# ridgeFull2DvsBL1D(string(path, "sim034/"))
# seamount(string(path, "sim035/"))
seamountFull2DvsBL(string(path, "sim035/"))