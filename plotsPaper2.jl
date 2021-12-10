using PyPlot, PyCall, Printf, HDF5, Dierckx

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

include("myJuliaLib.jl")

# for loading data
include("1dtc_pg/setup.jl")
include("2dpg/setup.jl")

# matplotlib
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

# utility functions
function boundaryCorrection(χI::Array{Float64,1}, z::Array{Float64,1}, q::Float64)
    A = -χI[1]
    χIz0 = differentiate_pointwise(χI[1:3], z[1:3], z[1], 1)
    B = -χIz0/q + A
    χB = @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
    return χB
end
function constructFullSolution(m::ModelSetup1DPG, s::ModelState1DPG, z::Array{Float64,1})
    # BL thickness
    q = (1/(4*m.ν[1])*(m.f^2/m.ν[1] + m.N2*tan(m.θ)^2/m.κ[1]))^(1/4)
    
    # interior vars
    bI = s.b
    χI = -differentiate(bI, m.z)*tan(m.θ).*m.ν/m.f^2

    # interpolate onto new grid 
    χI_fine = Spline1D(m.z .- m.z[1], χI)(z)
    bI_fine = Spline1D(m.z .- m.z[1], bI)(z)

    # BL correction
    χB = boundaryCorrection(χI_fine, z, q)
    bB = cumtrapz(χB*m.N2*tan(m.θ)/m.κ[1], z) .- trapz(χB*m.N2*tan(m.θ)/m.κ[1], z)

    # full sol
    χ = χI_fine + χB
    b = bI_fine + bB
    return χ, b
end
function constructFullSolution(m::ModelSetup2DPG, s::ModelState2DPG, z::Array{Float64,1}, ix::Int64)
    # interior vars
    bI = s.b[ix, :]
    χI = s.χ[ix, :]

    # BL thickness 
    bIξ = ξDerivative(m, s.b)
    q = (1/(4*m.ν[ix, 1])*(m.f^2/m.ν[ix, 1] - m.Hx[ix]*bIξ[ix, 1]/m.H[ix]/m.κ[ix, 1]))^(1/4)

    # interpolate onto new grid 
    χI_fine = Spline1D(m.z[ix, :] .- m.z[ix, 1], χI)(z)
    bI_fine = Spline1D(m.z[ix, :] .- m.z[ix, 1], bI)(z)

    # BL correction
    χB = boundaryCorrection(χI_fine, z, q)
    bB = cumtrapz(χB*bIξ[ix, 1]/m.κ[ix, 1], z) .- trapz(χB*bIξ[ix, 1]/m.κ[ix, 1], z) 

    # full sol
    χ = χI_fine + χB
    b = bI_fine + bB
    return χ, b
end
function BLtransport2D(m::ModelSetup2DPG, s::ModelState2DPG)
    dbdξ = ξDerivative(m, s.b[:, 1])
    μ = m.ν[1, 1] / m.κ[1, 1]
    return @. m.κ[:, 1]/m.Hx * μ*m.Hx/m.f^2 * dbdξ / (1 - μ*m.Hx/m.f^2 * dbdξ)
end
function exchangeVel2D(m::ModelSetup2DPG, χ::Array{Float64,1})
    if m.coords == "cartesian"
        # uσ = -dξ(χ)/H
        uσ = -ξDerivative(m, χ)./m.H
    elseif m.coords == "cylindrical"
        # uσ = -dρ(ρ*χ)/(H*ρ)
        uσ = -ξDerivative(m, m.ξ.*χ)./m.H./m.ξ
        # assume χ = 0 at ρ = 0
        fd_ξ = mkfdstencil([0, m.ξ[1], m.ξ[2]], m.ξ[1], 1)
        uσ[1] = -(fd_ξ[2]*m.ξ[1]*χ[1] + fd_ξ[3]*m.ξ[2]*χ[2])/(m.H[1]*m.ξ[1])
    end
    return m.H.*uσ
end
function get_pq(S, T, δ, μ)
    r = (-1 + im*sqrt(3))/2
    c = sqrt(μ^2*T^2/4 + (1 + μ*S)^3/27)
    λ = sqrt(2)/δ * sqrt(r*cbrt(-μ*T/2 + c) + conj(r)*cbrt(-μ*T/2 - c))
    q = real(λ)
    p = imag(λ)
    return p, q
end

# plotting functions
function slopeFull1DvsBL1D(folder)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 6.5/1.62), sharey=true)

    axins11 = inset_locator.inset_axes(ax[1, 1], width="50%", height="50%")
    axins21 = inset_locator.inset_axes(ax[2, 1], width="50%", height="50%")

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow $u^y$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification $N^2 + \partial_z b$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    fig.text(0.05, 0.98, L"$S = O(10^{-3})$:", ha="left", va="top")
    fig.text(0.05, 0.50, L"$S = O(10^{-1})$:", ha="left", va="top")

    subplots_adjust(hspace=0.5)

    # limits
    ax[1, 1].set_xlim([-0.05, 1.7])
    ax[2, 1].set_xlim([-0.5, 14])
    ax[1, 2].set_xlim([-2.5, 0.5])
    ax[2, 2].set_xlim([-22, 4])
    ax[1, 3].set_xlim([0, 1.3])
    ax[2, 3].set_xlim([0, 1.3])
    ax[1, 1].set_ylim([0, 2])
    ax[1, 2].set_ylim([0, 2])
    ax[1, 3].set_ylim([0, 2])
    ax[2, 1].set_ylim([0, 2])
    ax[2, 2].set_ylim([0, 2])
    ax[2, 3].set_ylim([0, 2])
    axins11.set_ylim([0, 0.1])
    axins21.set_ylim([0, 0.1])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 1D small S
        m = loadSetup1DPG(string(folder, "S_small/bl/setup.h5"))
        mFull = loadSetup1DPG(string(folder, "S_small/full/setup.h5"))
        s = loadState1DPG(string(folder, "S_small/bl/state$i.h5"))
        z = mFull.z .- mFull.z[1]
        χ, b = constructFullSolution(m, s, z)
        v = cumtrapz(m.f*(χ .- χ[end])./mFull.ν, z)
        Bz = m.N2 .+ differentiate(b, z)
        label = string(Int64(m.Δt*(s.i[1] - 1)/secsInYear), " years")
        ax[1, 1].plot(1e3*χ,   z/1e3, c=color, label=label)
        axins11.plot(1e3*χ,   z/1e3, c=color, label=label)
        ax[1, 2].plot(1e2*v,   z/1e3, c=color, label=label)
        ax[1, 3].plot(1e6*Bz,  z/1e3, c=color, label=label)

        # full 1D small S
        m = loadSetup1DPG(string(folder, "S_small/full/setup.h5"))
        s = loadState1DPG(string(folder, "S_small/full/state$i.h5"))
        z = m.z .- m.z[1]
        v = s.v
        Bz = m.N2 .+ differentiate(b, z)
        ax[1, 1].plot(1e3*χ,   z/1e3, "k:")
        axins11.plot(1e3*χ,   z/1e3, "k:")
        ax[1, 2].plot(1e2*v,   z/1e3, "k:")
        ax[1, 3].plot(1e6*Bz,  z/1e3, "k:")

        # BL 1D big S
        m = loadSetup1DPG(string(folder, "S_big/bl/setup.h5"))
        mFull = loadSetup1DPG(string(folder, "S_big/full/setup.h5"))
        s = loadState1DPG(string(folder, "S_big/bl/state$i.h5"))
        z = mFull.z .- mFull.z[1]
        χ, b = constructFullSolution(m, s, z)
        v = cumtrapz(m.f*(χ .- χ[end])./mFull.ν, z)
        Bz = m.N2 .+ differentiate(b, z)
        label = string(Int64(m.Δt*(s.i[1] - 1)/secsInYear), " years")
        ax[2, 1].plot(1e3*χ,   z/1e3, c=color, label=label)
        axins21.plot(1e3*χ,   z/1e3, c=color, label=label)
        ax[2, 2].plot(1e2*v,   z/1e3, c=color, label=label)
        ax[2, 3].plot(1e6*Bz,  z/1e3, c=color, label=label)

        # full 1D big S
        m = loadSetup1DPG(string(folder, "S_big/full/setup.h5"))
        s = loadState1DPG(string(folder, "S_big/full/state$i.h5"))
        z = m.z .- m.z[1]
        v = s.v
        Bz = m.N2 .+ differentiate(b, z)
        ax[2, 1].plot(1e3*χ,   z/1e3, "k:")
        axins21.plot(1e3*χ,   z/1e3, "k:")
        ax[2, 2].plot(1e2*v,   z/1e3, "k:")
        ax[2, 3].plot(1e6*Bz,  z/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL 1D", "full 1D"]
    ax[1, 2].legend(custom_handles, custom_labels, loc=(0.5, 0.4))
    ax[1, 3].legend(loc="upper left")
    
    savefig("slopeFull1DvsBL1D.pdf")
    println("slopeFull1DvsBL1D.pdf")
    plt.close()
end

function TFcoords()
    # params for ridge
    nξ = 2^8
    nσ = 2^8
    H0 = 2e3
    L = 2e6

    # TF grid
    ξ = collect(0:L/(nξ - 1):L)
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    ξξ = repeat(ξ, 1, nσ)
    σσ = repeat(σ', nξ, 1)

    # depth
    H = @. H0*(1 + 0.4*cos(2*π*ξ/L))

    # physical grid
    x = repeat(ξ, 1, nσ)
    z = repeat(σ', nξ, 1).*repeat(H, 1, nσ)

    # level sets 
    σlevels = -1.0:0.1:0.0
    ξlevels = 0:L/9:L

    # plot ξ and σ surfaces
    fig, ax = subplots()
    ax.fill_between(ξ, -H, -H0*1.4, color="k", alpha=0.3, lw=0.0)
    ax.contour(x, z, σσ, σlevels, colors="k", linestyles="-")
    ax.contour(x, z, ξξ, ξlevels, colors="k", linestyles="-")
    ax.axhline(0, lw=1, ls="-", c="k")
    ax.axvline(L, lw=1, ls="-", c="k")
    ax.set_xticks([])
    ax.set_yticks([])
	ax.spines["left"].set_visible(false)
	ax.spines["bottom"].set_visible(false)
    ax.set_ylim([-H0*1.4, 0])
    tight_layout()
    savefig("TFcoords.svg")
    println("TFcoords.svg")
end

function ridge(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    m = loadSetup2DPG(string(folder, "2dpg/mu1/setup.h5"))
    s = loadState2DPG(string(folder, "2dpg/mu1/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, 1e3*s.χ,  "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"); ax=ax[1])
    ridgePlot(m, s, 1e2*s.uη, "", string(L"along-ridge flow $u^y$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"); ax=ax[2], style="pcolormesh")
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

    axins = inset_locator.inset_axes(ax[1], width="50%", height="50%")

    ax[1].set_ylabel(L"$z$ (km)")
    ax[1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2].set_xlabel(string(L"along-ridge flow $u^y$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")

    subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # 2D setup
    m2D = loadSetup2DPG(string(folder, "2dpg/mu1/setup.h5"))
    ix= argmin(abs.(m2D.x[:, 1] .- m2D.L/4))
    m1D = loadSetup1DPG(string(folder, "bl1d/mu1/setup.h5"))

    # limits
    ax[1].set_xlim([-0.05, 1.7])
    ax[2].set_xlim([-2.5, 0.5])
    ax[3].set_xlim([0, 1.3])
    axins.set_ylim([m2D.z[ix, 1]/1e3, (m2D.z[ix, 1] + 55)/1e3])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))


    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 1D
        s = loadState1DPG(string(folder, "bl1d/mu1/state$i.h5"))
        z = m2D.z[ix, :] .- m2D.z[ix, 1]
        χ, b = constructFullSolution(m1D, s, z)
        v = cumtrapz(m1D.f*(χ .- χ[end])./m2D.ν[ix, :], z)
        Bz = m1D.N2 .+ differentiate(b, z)
        label = string(Int64(m1D.Δt*(s.i[1] - 1)/secsInYear), " years")
        ax[1].plot(1e3*χ,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        axins.plot(1e3*χ,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        ax[2].plot(1e2*v,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        ax[3].plot(1e6*Bz,  z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)

        # full 2D
        s = loadState2DPG(string(folder, "2dpg/mu1/state$i.h5"))
        bz = differentiate(s.b[ix, :], m2D.z[ix, :])
        ax[1].plot(1e3*s.χ[ix, :],   m2D.z[ix, :]/1e3, "k:")
        axins.plot(1e3*s.χ[ix, :],   m2D.z[ix, :]/1e3, "k:")
        ax[2].plot(1e2*s.uη[ix, :],  m2D.z[ix, :]/1e3, "k:")
        ax[3].plot(1e6*bz,           m2D.z[ix, :]/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls="-", lw="1"),
                      lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL 1D", "full 2D"]
    ax[2].legend(custom_handles, custom_labels)
    ax[3].legend()
    
    savefig("ridgeFull2DvsBL1D.pdf")
    println("ridgeFull2DvsBL1D.pdf")
    plt.close()
end

function seamount(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "L200km/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, 1e2*s.χ, "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-2}$ m$^2$ s$^{-1}$)"); ax=ax[1])
    ridgePlot(m, s, 1e1*s.uη, "", string(L"along-slope flow $u^y$", "\n", L"($\times 10^{-1}$ m s$^{-1}$)"); ax=ax[2], style="pcolormesh")
    ax[1].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[2].plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")

    # subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

    savefig("seamount.pdf")
    println("seamount.pdf")
    plt.close()
end

function seamountFull2DvsBL(folder)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 6.5/1.62), sharey=true)

    axins11 = inset_locator.inset_axes(ax[1, 1], width="100%", height="100%", 
                                        bbox_to_anchor=(0.3, 0.4, .5, .5), bbox_transform=ax[1, 1].transAxes, loc=3)
    axins21 = inset_locator.inset_axes(ax[2, 1], width="100%", height="100%", 
                                        bbox_to_anchor=(0.3, 0.4, .5, .5), bbox_transform=ax[2, 1].transAxes, loc=3)

    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-2}$ m$^2$ s$^{-1}$)"))
    ax[2, 2].set_xlabel(string(L"along-ridge flow $u^y$", "\n", L"($\times 10^{-1}$ m s$^{-1}$)"))
    ax[2, 3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    ax[1, 1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", (-0.04, 1.05), xycoords="axes fraction")

    fig.text(0.05, 0.98, "BL 1D:", ha="left", va="top")
    fig.text(0.05, 0.50, "BL 2D:", ha="left", va="top")

    subplots_adjust(hspace=0.5)

    # model setups
    m1DBL = loadSetup1DPG(string(folder, "bl1D/setup.h5"))
    m2D = loadSetup2DPG(string(folder, "full2D/setup.h5"))
    ix = argmin(abs.(m2D.x[:, 1] .- m2D.L/4))
    m2DBL = loadSetup2DPG(string(folder, "bl2D/setup.h5"))
    ixBL = argmin(abs.(m2DBL.x[:, 1] .- m2DBL.L/4))

    # limits
    ax[1, 1].set_xlim([-1.7, 0.05])
    ax[2, 1].set_xlim([-1.7, 0.05])
    axins11.set_xlim([-1.7, 0.05])
    axins21.set_xlim([-1.7, 0.05])
    ax[1, 2].set_xlim([-1.0, 4.0])
    ax[2, 2].set_xlim([-1.0, 4.0])
    ax[1, 3].set_xlim([0, 1.3])
    ax[2, 3].set_xlim([0, 1.3])
    ax[1, 1].set_ylim([m2D.z[ix, 1]/1e3, 0])
    ax[1, 2].set_ylim([m2D.z[ix, 1]/1e3, 0])
    ax[1, 3].set_ylim([m2D.z[ix, 1]/1e3, 0])
    ax[2, 1].set_ylim([m2D.z[ix, 1]/1e3, 0])
    ax[2, 2].set_ylim([m2D.z[ix, 1]/1e3, 0])
    ax[2, 3].set_ylim([m2D.z[ix, 1]/1e3, 0])
    axins11.set_ylim([m2D.z[ix, 1]/1e3, (m2D.z[ix, 1] + 55)/1e3])
    axins21.set_ylim([m2D.z[ix, 1]/1e3, (m2D.z[ix, 1] + 55)/1e3])

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # plot data
    for i=1:5
        # line color
        color = colors[i, :]

        # BL 1D
        m = m1DBL
        s = loadState1DPG(string(folder, "bl1D/state$i.h5"))
        z = m2D.z[ix, :] .- m2D.z[ix, 1]
        χ, b = constructFullSolution(m, s, z)
        v = cumtrapz(m.f*(χ .- χ[end])./m2D.ν[ix, :], z)
        Bz = m.N2 .+ differentiate(b, z)
        label = string(Int64(m.Δt*(s.i[1] - 1)/secsInYear), " years")
        ax[1, 1].plot(1e2*χ,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        axins11.plot(1e2*χ,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        ax[1, 2].plot(1e1*v,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        ax[1, 3].plot(1e6*Bz,  z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)

        # full 2D
        m = m2D
        s = loadState2DPG(string(folder, "full2D/state$i.h5"))
        bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax[1, 1].plot(1e2*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        axins11.plot(1e2*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        ax[1, 2].plot(1e1*s.uη[ix, :],  m.z[ix, :]/1e3, "k:")
        ax[1, 3].plot(1e6*bz,           m.z[ix, :]/1e3, "k:")

        # BL 2D
        m = m2DBL
        s = loadState2DPG(string(folder, "bl2D/state$i.h5"))
        z = m2D.z[ix, :] .- m2D.z[ix, 1]
        χ, b = constructFullSolution(m, s, z, ixBL)
        v = cumtrapz(m.f*(χ .- χ[end])./m2D.ν[ix, :], z)
        bz = differentiate(b, z)
        ax[2, 1].plot(1e2*χ,  z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        axins21.plot(1e2*χ,   z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        ax[2, 2].plot(1e1*v,  z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)
        ax[2, 3].plot(1e6*bz, z/1e3 .+ m2D.z[ix, 1]/1e3, c=color, label=label)

        # full 2D
        m = m2D
        s = loadState2DPG(string(folder, "full2D/state$i.h5"))
        bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax[2, 1].plot(1e2*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        axins21.plot(1e2*s.χ[ix, :],   m.z[ix, :]/1e3, "k:")
        ax[2, 2].plot(1e1*s.uη[ix, :],  m.z[ix, :]/1e3, "k:")
        ax[2, 3].plot(1e6*bz,           m.z[ix, :]/1e3, "k:")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["full 2D"]
    ax[1, 2].legend(custom_handles, custom_labels)
    ax[1, 3].legend()
    
    savefig("seamountFull2DvsBL.pdf")
    println("seamountFull2DvsBL.pdf")
    plt.close()
end

function ridgeN2exp(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    m = loadSetup2DPG(string(folder, "const/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "const/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, 1e3*s.χ,  "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"); ax=ax[1])

    m = loadSetup2DPG(string(folder, "exp/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "exp/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, 1e3*s.χ,  "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"); ax=ax[2])

    ax[1].annotate(L"(a) $N^2 =$ constant", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate(L"(b) $N^2 \sim \exp(z/\delta)$", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")

    savefig("ridgeN2exp.pdf")
    println("ridgeN2exp.pdf")
    plt.close()
end
# function expStrat(folder)
#     fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2)) 

#     ax[1].set_xlabel(L"$\xi$ (km)")
#     ax[1].set_ylabel(string("BL transport\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))

#     ax[2].set_xlabel(L"$\xi$ (km)")
#     ax[2].set_ylabel(string(L"exchange velocity $H u^\sigma$", "\n", L"($\times 10^{-6}$ m s$^{-1}$)"))

#     ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
#     ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")

#     # color map
#     colors = pl.cm.viridis(range(1, 0, length=5))

#     mConst = loadSetup2DPG(string(folder, "/const/bl2D/setup.h5"))
#     mExp   = loadSetup2DPG(string(folder, "/exp/bl2D/setup.h5"))
#     for i=0:5
#         if i == 0
#             c = "tab:red"
#         else
#             c = colors[i, :]
#         end
#         s = loadState2DPG(string(folder, "/const/bl2D/state$i.h5"))
#         χtheory = BLtransport2D(mConst, s)
#         W = exchangeVel2D(mConst, χtheory)
#         label = string(Int64((s.i[1] - 1)*mConst.Δt/86400/360), " years")
#         ax[1].plot(mConst.ξ/1e3, 1e3*χtheory, c=c, label=string(L"$N^2 = $", "const."))
#         ax[2].plot(mConst.ξ/1e3, 1e6*W, c=c, label=label)

#         s = loadState2DPG(string(folder, "/exp/bl2D/state$i.h5"))
#         χtheory = BLtransport2D(mExp, s)
#         W = exchangeVel2D(mExp, χtheory)
#         ax[1].plot(mExp.ξ/1e3, 1e3*χtheory, c=c, ls="--", label=L"$N^2 \sim \exp(z/\delta)$")
#         ax[2].plot(mExp.ξ/1e3, 1e6*W, c=c, ls="--")
#     end

#     ax[1].set_xlim([0, mConst.L/1e3])
#     # ax[1].set_ylim([-20, 0])

#     ax[2].set_xlim([0, mConst.L/1e3])
#     # ax[2].set_ylim([-1, 20])

#     # ax[1].legend()
#     ax[2].legend()

#     tight_layout()

#     savefig("expStrat.pdf")
#     println("expStrat.pdf")
#     plt.close()
# end
function transportAndExchange(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2)) 

    ax[1].set_xlabel(L"$x$ (km)")
    ax[1].set_ylabel(string("BL transport\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))

    ax[2].set_xlabel(L"$x$ (km)")
    ax[2].set_ylabel(string(L"exchange velocity $H u^\sigma$", "\n", L"($\times 10^{-9}$ m s$^{-1}$)"))

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")

    mConst = loadSetup2DPG(string(folder, "/const/bl2D/setup.h5"))
    mExp   = loadSetup2DPG(string(folder, "/exp/bl2D/setup.h5"))

    s = loadState2DPG(string(folder, "/const/bl2D/state1.h5"))
    χtheory = BLtransport2D(mConst, s)
    W = exchangeVel2D(mConst, χtheory)
    ax[1].plot(mConst.ξ/1e3, 1e3*χtheory, label=string(L"$N^2 = $", "const."))
    ax[2].plot(mConst.ξ/1e3, 1e9*W)

    s = loadState2DPG(string(folder, "/exp/bl2D/state1.h5"))
    χtheory = BLtransport2D(mExp, s)
    W = exchangeVel2D(mExp, χtheory)
    ax[1].plot(mExp.ξ/1e3, 1e3*χtheory,label=L"$N^2 \sim \exp(z/\delta)$")
    ax[2].plot(mExp.ξ/1e3, 1e9*W)

    ax[1].set_xlim([0, mConst.L/1e3])
    ax[1].set_ylim([-3, 3])

    ax[2].set_xlim([0, mConst.L/1e3])
    ax[2].set_ylim([-8, 18])

    ax[1].legend()

    subplots_adjust(wspace=0.3)

    savefig("transportAndExchange.pdf")
    println("transportAndExchange.pdf")
    plt.close()
end

function pq()
    δ = 10
    μ = 1
    Ss = 10. .^(-3:0.1:1)
    Ts = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    cs = pl.cm.viridis(range(1, 0, length=size(Ts, 1)))

    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

    ax[1].set_xlim([Ss[1], Ss[end]])
    ax[2].set_xlim([Ss[1], Ss[end]])
    ax[1].set_ylim([0.5, 3.0])

    for i=1:size(Ts, 1)
        T = Ts[i]
        c = cs[i, :] 
        label = latexstring("\$T = 10^", @sprintf("{%.0f}\$", log10(T)))
        pq = get_pq.(Ss, T, δ, μ)
        ax[1].semilogx(Ss, δ*last.(pq),  c=c, ls="-", label=label)
        ax[2].semilogx(Ss, δ*first.(pq), c=c, ls="-", label=label)
    end

    # T = 0 
    ax[1].semilogx(Ss, (1 .+ μ.*Ss).^(1/4), "k:")
    ax[2].semilogx(Ss, (1 .+ μ.*Ss).^(1/4), "k:")

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = [L"1D and 2D theory ($T = 0$)"]
    ax[1].legend(custom_handles, custom_labels)
    ax[2].legend(loc=(0.05, 0.45))
    ax[1].set_xlabel(L"slope Burger number $S$")
    ax[2].set_xlabel(L"slope Burger number $S$")
    ax[1].set_ylabel(L"decay scale $\delta q$")
    ax[2].set_ylabel(L"oscillation scale $\delta p$")
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    # ax[2].annotate(L"$\mu = 1$", (0.8, 0.9), xycoords="axes fraction")
    savefig("pq.pdf")
    println("pq.pdf")
end

path = "../sims/"

# slopeFull1DvsBL1D(string(path, "sim044/"))
# TFcoords()
# ridge(string(path, "sim039/"))
# ridgeFull2DvsBL1D(string(path, "sim039/"))
# seamount(string(path, "sim035/"))
# seamountFull2DvsBL(string(path, "sim042/"))
# ridgeN2exp(string(path, "sim037/"))
# transportAndExchange(string(path, "sim037/"))
pq()