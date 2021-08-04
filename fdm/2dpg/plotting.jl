################################################################################
# Functions useful for plotting
################################################################################

pl = pyimport("matplotlib.pylab")

"""
    ax = ridgePlot(m, s, field, titleString, cbarLabel; ax, vext, cmap)

Create 2D plot of `field` with isopycnals given by the buoyancy perturbation `b`
from the model state `s`. Set the title to `titleString` and colorbar label to 
`cbarLabel`. Return the axis handle `ax`.

Optional: 
    - provide `ax`
    - set the vmin/vmax manually with `vext`
    - set different colormap `cmap`
"""
function ridgePlot(m::ModelSetup, s::ModelState, field::Array{Float64,2}, titleString::AbstractString, cbarLabel::AbstractString; ax=nothing, vext=nothing, cmap="RdBu_r")
    # km
    xx = m.x/1000
    zz = m.z/1000

    if ax === nothing
        fig, ax = subplots(1)
    end

    # set min and max
    if vext === nothing
        vmax = maximum(abs.(field))
        vmin = -vmax
        extend = "neither"
    else
        vmax = vext
        vmin = -vext
        extend = "both"
    end

    # regular min and max for viridis
    if cmap == "viridis"
        vmin = minimum(field)
        vmax = maximum(field)
        extend = "neither"
    end

    # 2D plot
    img = ax.pcolormesh(xx, zz, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    cb = colorbar(img, ax=ax, label=cbarLabel, extend=extend)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)

    # isopycnal contours
    nLevels = 20
    lowerLevel = m.N^2*minimum(m.z)
    upperLevel = 0
    levels = lowerLevel:(upperLevel - lowerLevel)/(nLevels - 1):upperLevel
    ax.contour(xx, zz, s.b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)

    # ridge shading
    ax.fill_between(xx[:, 1], zz[:, 1], minimum(zz), color="k", alpha=0.3, lw=0.0)

    # labels
    ax.set_title(titleString)
    ax.set_xlabel(L"$x$ (km)")
    ax.set_ylabel(L"$z$ (km)")
    ax.set_xticks([0, 500, 1000, 1500, 2000])

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)

    tight_layout()
    
    return ax
end

"""
    profilePlot(datafiles, iξ)

Plot profiles of χ, v, b_z from HDF5 snapshot files of buoyancy in the `datafiles` list
at ξ = ξ[iξ].
"""
function profilePlot(setupFile, stateFiles, iξ)
    # ModelSetup 
    m = loadSetup2DPG(setupFile)

    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 2), sharey=true)

    ax[1].set_xlabel(string("streamfunction,\n", L"$\chi$ (m$^2$ s$^{-1}$)"))
    ax[1].set_ylabel(L"$z$ (km)")

    ax[2].set_xlabel(string("along-ridge vel.,\n", L"$v$ (m s$^{-1}$)"))

    ax[3].set_xlabel(string("stratification,\n", L"$\partial_z b$ (s$^{-2}$)"))

    subplots_adjust(bottom=0.3, top=0.90, left=0.1, right=0.95, wspace=0.2, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # plot data from `datafiles`
    for i=1:size(stateFiles, 1)
        # load
        s = loadCheckpoint2DPG(stateFiles[i])
        u, v, w = transformFromTF(m, s)

        # stratification
        bz = zDerivativeTF(m, s.b)

        # colors and labels
        label = string(Int64(round(s.i[1]*m.Δt/secsInYear)), " years")
        color = colors[i, :]

        # plot
        ax[1].plot(s.χ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
        ax[2].plot(v[iξ, :],   m.z[iξ, :]/1e3, c=color)
        ax[3].plot(bz[iξ, :],  m.z[iξ, :]/1e3, c=color)
    end

    ax[1].legend()

    savefig("profiles.png")
    println("profiles.png")
end

"""
    plotCurrentState(m, s, iImg)

Make some ridge plots of the current model state using the label number `iImg`.
"""
function plotCurrentState(m::ModelSetup, s::ModelState, iImg::Int64)
    # convert to physical coordinates 
    u, v, w = transformFromTF(m, s)

    # plots
    ridgePlot(m, s, s.χ, @sprintf("t = %4d years", s.i[1]*m.Δt/secsInYear), L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)")
    savefig(@sprintf("chi%03d.png", iImg))
    close()

    ridgePlot(m, s, s.b, @sprintf("t = %4d years", s.i[1]*m.Δt/secsInYear), L"buoyancy, $b$ (m s$^{-2}$)")
    savefig(@sprintf("b%03d.png", iImg))
    close()

    ridgePlot(m, s, v, @sprintf("t = %4d years", s.i[1]*m.Δt/secsInYear), L"along-ridge velocity, $v$ (m s$^{-1}$)")
    savefig(@sprintf("v%03d.png", iImg))
    close()
end
