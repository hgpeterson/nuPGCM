
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

function ridge(folder)
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62), sharey=true)

    # big L
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

    # little L
    # m = loadSetup2DPG(string(folder, "L100km/full2D/setup.h5"))
    # s = loadState2DPG(string(folder, "L100km/full2D/state1.h5"))
    m = loadSetup2DPG(string(folder, "L120km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L120km/full2D/state1.h5"))
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

    savefig("ridge.pdf")
    println("ridge.pdf")
    plt.close()
end

function full2DvsBL1D(folder)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 6.5/1.62), sharey=true)

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"(m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow $v$", "\n", L"(m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"(s$^{-2}$)"))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    mBig = loadSetup2DPG(string(folder, "L2000km/full2D/setup.h5"))
    fig.text(0.05, 0.98, string(L"$L = $", Int64(mBig.L/1e3), " km:"), ha="left", va="top")
    ix1 = argmin(abs.(mBig.x[:, 1] .- mBig.L/4))

    # m = loadSetup2DPG(string(folder, "L100km/full2D/setup.h5"))
    mSmall = loadSetup2DPG(string(folder, "L120km/full2D/setup.h5"))
    fig.text(0.05, 0.52, string(L"$L = $", Int64(mSmall.L/1e3), " km:"), ha="left", va="top")
    ix2 = argmin(abs.(mSmall.x[:, 1] .- mSmall.L/4))

    # limits
    # ax[1, 1].set_ylim([mBig.z[ix1, 1]/1e3,   (mBig.z[ix1, 1] + 100)/1e3])
    # ax[2, 1].set_ylim([mSmall.z[ix2, 1]/1e3, (mSmall.z[ix1, 1] + 100)/1e3])

    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 1D: big L
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
        ax[1, 1].plot(χ,   ẑ*cos(c.θ)/1e3 .+ mBig.z[ix1, 1]/1e3, c=color, label=label)
        ax[1, 2].plot(v,   ẑ*cos(c.θ)/1e3 .+ mBig.z[ix1, 1]/1e3, c=color, label=label)
        ax[1, 3].plot(Bz,  ẑ*cos(c.θ)/1e3 .+ mBig.z[ix1, 1]/1e3, c=color, label=label)

        # full 2D: big L 
        s = loadState2DPG(string(folder, "L2000km/full2D/state$i.h5"))
        bz = differentiate(s.b[ix1, :], mBig.z[ix1, :])
        ax[1, 1].plot(s.χ[ix1, :],   mBig.z[ix1, :]/1e3, "k:")
        ax[1, 2].plot(s.uη[ix1, :],  mBig.z[ix1, :]/1e3, "k:")
        ax[1, 3].plot(bz,            mBig.z[ix1, :]/1e3, "k:")

        # BL 1D: little L
        # c = loadCheckpoint1DTCPG(string(folder, "L100km/bl1D/checkpoint$i.h5"))
        c = loadCheckpoint1DTCPG(string(folder, "L120km/bl1D/checkpoint$i.h5"))
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
        ax[2, 1].plot(χ,   ẑ*cos(c.θ)/1e3 .+ mSmall.z[ix2, 1]/1e3, c=color, label=label)
        ax[2, 2].plot(v,   ẑ*cos(c.θ)/1e3 .+ mSmall.z[ix2, 1]/1e3, c=color, label=label)
        ax[2, 3].plot(Bz,  ẑ*cos(c.θ)/1e3 .+ mSmall.z[ix2, 1]/1e3, c=color, label=label)

        # full 2D: little L
        # s = loadState2DPG(string(folder, "L100km/full2D/state$i.h5"))
        s = loadState2DPG(string(folder, "L120km/full2D/state$i.h5"))
        bz = differentiate(s.b[ix2, :], mSmall.z[ix2, :])
        ax[2, 1].plot(s.χ[ix2, :],   mSmall.z[ix2, :]/1e3, "k:")
        ax[2, 2].plot(s.uη[ix2, :],  mSmall.z[ix2, :]/1e3, "k:")
        ax[2, 3].plot(bz,            mSmall.z[ix2, :]/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL 1D", "full 2D"]
    ax[1, 2].legend(custom_handles, custom_labels)
    ax[1, 1].legend()
    
    savefig("full2DvsBL1D.pdf")
    println("full2DvsBL1D.pdf")
    plt.close()
end
function boundaryCorrection(χI::Array{Float64,1}, z::Array{Float64,1}, q::Float64)
    A = -χI[1]
    χIz0 = differentiate_pointwise(χI[1:3], z[1:3], z[1], 1)
    B = -χIz0/q + A
    χB = @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
    return χB
end

function full2DvsBL2D(folder)
    # L = "L120km"
    L = "L200km"

    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 6.5/1.62/2), sharey=true)

    ax[1].set_ylabel(L"$z$ (km)")
    ax[1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"(m$^2$ s$^{-1}$)"))
    ax[2].set_xlabel(string(L"along-ridge flow $v$", "\n", L"(m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"(s$^{-2}$)"))

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    mBL = loadSetup2DPG(string(folder, L, "/bl2D/setup.h5"))
    mFull = loadSetup2DPG(string(folder, L, "/full2D/setup.h5"))
    ixBL = argmin(abs.(mBL.x[:, 1] .- mBL.L/4))
    ixFull = argmin(abs.(mFull.x[:, 1] .- mFull.L/4))
    println(mBL.x[ixBL, 1])
    println(mFull.x[ixFull, 1])

    # limits
    # ax[1].set_ylim([mBL.z[ix, 1]/1e3, (mBL.z[ix, 1] + 100)/1e3])

    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 2D
        s = loadState2DPG(string(folder, L, "/bl2D/state$i.h5"))
        bI = s.b[ixBL, :]
        χI = s.χ[ixBL, :]
        bIξ = ξDerivative(mBL, s.b)
        q = (1/(4*mBL.ν[ixBL, 1])*(mBL.f^2/mBL.ν[ixBL, 1] - mBL.Hx[ixBL]*bIξ[ixBL, 1]/mBL.H[ixBL]/mBL.κ[ixBL, 1]))^(1/4)
        χB = boundaryCorrection(χI, mBL.z[ixBL, :] .- mBL.z[ixBL, 1], q)
        bB = cumtrapz(χB*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) .- trapz(χB.*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) 
        χ = χI + χB
        b = bI + bB
        bz = differentiate(b, mBL.z[ixBL, :])
        label = string(Int64(round(s.i[1]*mBL.Δt/86400/360)), " years")
        ax[1].plot(χ,            mBL.z[ixBL, :]/1e3, c=color, label=label)
        ax[2].plot(s.uη[ixBL, :],  mBL.z[ixBL, :]/1e3, c=color, label=label)
        ax[3].plot(bz,           mBL.z[ixBL, :]/1e3, c=color, label=label)

        # full 2D
        s = loadState2DPG(string(folder, L, "/full2D/state$i.h5"))
        bz = differentiate(s.b[ixFull, :], mFull.z[ixFull, :])
        ax[1].plot(s.χ[ixFull, :],   mFull.z[ixFull, :]/1e3, "k:")
        ax[2].plot(s.uη[ixFull, :],  mFull.z[ixFull, :]/1e3, "k:")
        ax[3].plot(bz,           mFull.z[ixFull, :]/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL 2D", "full 2D"]
    ax[2].legend(custom_handles, custom_labels)
    ax[1].legend()
    
    savefig("full2DvsBL2D.pdf")
    println("full2DvsBL2D.pdf")
    plt.close()
end

function seamount(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L200km/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1])
    ridgePlot(m, s, s.uη, "", L"along-ridge flow $v$ (m s$^{-1}$)"; ax=ax[2])
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

function seamountWMT(folder)
    fig, ax = subplots(1, 2, figsize=(3.404, 3.404/1.62), sharey=true)
    ax[1].set_xlabel(string(L"int. buoyancy flux $F$", "\n", L"($\times 10^{-5}$ m$^3$ s$^{-3}$)"))
    ax[1].set_ylabel(L"buoyancy b ($\times 10^{-3}$ m s$^{-2}$)")
    ax[2].set_xlabel(string(L"net diapyc. flow $\mathcal{E}_\mathregular{net}$", "\n", L"($\times 10^{-1}$ m$^2$ s$^{-1}$)"))
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")

    # for a in ax
    #     a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    # end

    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))

    colors = pl.cm.viridis(range(1, 0, length=5))
    for i=1:5
        println(i)
        s = loadState2DPG(string(folder, "L200km/full2D/state$i.h5"))

        # downward diffusive buoyancy flux as a function of ξ and σ
        B = m.κ.*σDerivative(m, s.b)./repeat(m.H, 1, m.nσ)

        # interpolate 
        B_spl = Spline2D(m.ξ, m.σ, B)

        # get σ as a function of ξ and b
        function σ(ξ::Float64, b::Float64)
            j = argmin(m.ξ .- ξ)
            b_profile = s.b[j, :]
            if b <= minimum(b_profile)
                return -1
            elseif b >= maximum(b_profile)
                return 0
            else
                spl = Spline1D(m.σ, b_profile .- b)
                return roots(spl)[1]
            end
        end

        # B as function of ξ and b
        B(ξ::Float64, b::Float64) = evaluate(B_spl, ξ, σ(ξ, b))

        # F(b) = ∫ B(ξ, b) dξ
        # bmin = minimum(s.b)
        # bmax = maximum(s.b)
        bmin = -2.5e-3
        bmax = -0.5e-3
        n = 2^9
        b = range(bmin, bmax; length=n)
        F = zeros(n)
        for k=1:n
            F[k] = trapz(B.(m.ξ, b[k]), m.ξ)
        end

        # E_net = ∂F/∂b
        E_net = differentiate(F, b[2] - b[1])

        # plot
        label = string(Int64(round(s.i[1]*m.Δt/86400/360)), " years")
        ax[1].plot(1e5*F,     1e3*b, c=colors[i, :], label=label)
        ax[2].plot(1e1*E_net, 1e3*b, c=colors[i, :], label=label)
    end
    ax[2].legend()
    tight_layout()

    savefig("seamountWMT.pdf")
    println("seamountWMT.pdf")
    plt.close()
end

path = "../sims/"

# ridge(string(path, "sim034/"))
# full2DvsBL1D(string(path, "sim034/"))
# full2DvsBL2D(string(path, "sim034/"))
# seamount(string(path, "sim035/"))
# full2DvsBL2D(string(path, "sim035/"))
seamountWMT(string(path, "sim035/"))