################################################################################
# Functions useful for plotting
################################################################################

pl = pyimport("matplotlib.pylab")

"""
    ax = ridgePlot(field, b, titleString, cbarLabel; vext)

Create 2D plot of `field` with isopycnals given by the buoyancy perturbation `b`.
Set the title to `titleString` and colorbar label to `cbarLabel`. Return the axis 
handle `ax`.

Optional: set the vmin/vmax manually with vext.
"""
function ridgePlot(field, b, titleString, cbarLabel; vext=nothing, cmap="RdBu_r")
    # full buoyancy for isopycnals
    B = N^2*zz + b 

    fig, ax = subplots(1)

    # set min and max
    if vext == nothing
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
    img = ax.pcolormesh(xx/1000, zz, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true)
    cb = colorbar(img, ax=ax, label=cbarLabel, extend=extend)
    cb.ax.ticklabel_format(style="sci", scilimits=(-3, 3))

    # isopycnal contours
    nLevels = 20
    lowerLevel = N^2*minimum(zz)
    upperLevel = 0
    levels = lowerLevel:(upperLevel - lowerLevel)/(nLevels - 1):upperLevel
    ax.contour(xx/1000, zz, B, levels=levels, colors="k", alpha=0.3, linestyles="-")

    # ridge shading
    ax.fill_between(xx[:, 1]/1000, zz[:, 1], minimum(zz), color="k", alpha=0.3)

    # labels
    ax.set_title(titleString)
    ax.set_xlabel(L"$x$ (km)")
    ax.set_ylabel(L"$z$ (m)")

    tight_layout()
    
    return ax
end

"""
    profilePlot(datafiles, ix)

Plot profiles of b, u, v, w from HDF5 snapshot files of buoyancy in the `datafiles` list
at x = x[ix].
"""
function profilePlot(datafiles, ix)
    # init plot
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62), sharey=true)

    ax[1, 1].set_xlabel(L"$u$ (m s$^{-1}$)")
    ax[1, 1].set_ylabel(L"$z$ (m)")
    ax[1, 1].set_title("cross-ridge velocity")

    ax[1, 2].set_xlabel(L"$v$ (m s$^{-1}$)")
    ax[1, 2].set_title("along-ridge velocity")

    ax[2, 1].set_xlabel(L"$w$ (m s$^{-1}$)")
    ax[2, 1].set_ylabel(L"$z$ (m)")
    ax[2, 1].set_title("vertical velocity")

    ax[2, 2].set_xlabel(L"$B_z$ (s$^{-2}$)")
    ax[2, 2].set_title("stratification")

    tight_layout()

    ax[1, 1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax[1, 2].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax[2, 1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax[2, 2].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # left-hand side for inversion equations
    inversionLHS = lu(getInversionLHS())

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # plot data from `datafiles`
    for i=1:size(datafiles, 1)
        if occursin("b", datafiles[i])
            # fixed 1D solution data files only need buoyancy
            file = h5open(datafiles[i], "r")
            b = read(file, "b")
            t = read(file, "t")
            close(file)

            # invert buoyancy for flow
            chi, û, v = invert(b, inversionLHS)
        elseif occursin("sol", datafiles[i])
            # canonical 1D solution data files have all variables needed
            file = h5open(datafiles[i], "r")
            b = read(file, "b")
            û = read(file, "û")
            v = read(file, "v")
            t = read(file, "t")
            close(file)
        end

        # convert to physical coordinates 
        u, w = rotate(û)

        # stratification #FIXME is this right???
        Bz = N^2*cosθ .+ ẑDerivative(b)

        # colors and labels
        if t == Inf
            label = "Steady State"
            c = "k"
        else
            label = string("Day ", Int64(round(t/86400)))
            c = colors[i, :]
        end

        # plot
        ax[1, 1].plot(u[ix, :],  zz[ix, :], c=c, label=label)
        ax[1, 2].plot(v[ix, :],  zz[ix, :], c=c)
        ax[2, 1].plot(w[ix, :],  zz[ix, :], c=c)
        ax[2, 2].plot(Bz[ix, :], zz[ix, :], c=c)
    end

    ax[1, 1].legend()

    savefig("profiles.png")
end

"""
    plotCurrentState(t, chi, û, v, b, iImg)

Plot the buoyancy and velocity state of the model at time `t` using label number `iImg`.
"""
function plotCurrentState(t, chi, û, v, b, iImg)
    # convert to physical coordinates 
    u, w = rotate(û)

    # plots
    ridgePlot(chi, b, @sprintf("streamfunction at t = %4d days", t/86400), L"$\chi$ (m$^2$ s$^{-1}$)")
    savefig(@sprintf("chi%03d.png", iImg))
    close()

    ridgePlot(b, b, @sprintf("buoyancy perturbation at t = %4d days", t/86400), L"$b$ (m s$^{-2}$)")
    savefig(@sprintf("b%03d.png", iImg))
    close()

    ridgePlot(u, b, @sprintf("cross-ridge velocity at t = %4d days", t/86400), L"$u$ (m s$^{-1}$)")
    savefig(@sprintf("u%03d.png", iImg))
    close()

    ridgePlot(v, b, @sprintf("along-ridge velocity at t = %4d days", t/86400), L"$v$ (m s$^{-1}$)")
    savefig(@sprintf("v%03d.png", iImg))
    close()

    ridgePlot(w, b, @sprintf("vertical velocity at t = %4d days", t/86400), L"$w$ (m s$^{-1}$)")
    savefig(@sprintf("w%03d.png", iImg))
    close()
end
function plotCurrentState(t, û, v, b, iImg)
    # convert to physical coordinates 
    u, w = rotate(û)

    ridgePlot(b, b, @sprintf("buoyancy perturbation at t = %4d days", t/86400), L"$b$ (m s$^{-2}$)")
    savefig(@sprintf("b%03d.png", iImg))
    close()

    ridgePlot(u, b, @sprintf("cross-ridge velocity at t = %4d days", t/86400), L"$u$ (m s$^{-1}$)")
    savefig(@sprintf("u%03d.png", iImg))
    close()

    ridgePlot(v, b, @sprintf("along-ridge velocity at t = %4d days", t/86400), L"$v$ (m s$^{-1}$)")
    savefig(@sprintf("v%03d.png", iImg))
    close()

    ridgePlot(w, b, @sprintf("vertical velocity at t = %4d days", t/86400), L"$w$ (m s$^{-1}$)")
    savefig(@sprintf("w%03d.png", iImg))
    close()
end
