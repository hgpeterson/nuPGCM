pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")

"""
    profilePlot(datafiles)

Plot profiles of b, χ, û, and v̂ from HDF5 snapshot files of buoyancy in the `datafiles` list.
"""
function profilePlot(datafiles)
    # init plot
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62))

    # insets
    axins21 = inset_locator.inset_axes(ax[2, 1], width="40%", height="40%")

    ax[1, 1].set_xlabel(L"$B_z$ (s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$z$ (m)")
    ax[1, 1].set_title("stratification")

    ax[1, 2].set_xlabel(L"$\chi$ (m$^2$ s$^{-1}$)")
    ax[1, 2].set_ylabel(L"$z$ (m)")
    ax[1, 2].set_title("streamfunction")

    ax[2, 1].set_xlabel(L"$u$ (m s$^{-1}$)")
    ax[2, 1].set_ylabel(L"$z$ (m)")
    ax[2, 1].set_title("cross-ridge velocity")

    ax[2, 2].set_xlabel(L"$v$ (m s$^{-1}$)")
    ax[2, 2].set_ylabel(L"$z$ (m)")
    ax[2, 2].set_title("along-ridge velocity")

    subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end
    axins21.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)))

    # zoomed z
    ax[2, 1].set_ylim([z[1]/1000, (z[1] + 200)/1000])

    # plot data from `datafiles`
    for i=1:size(datafiles, 1)
        # load
        c = loadCheckpointRot(datafiles[i])

        # convert to physical coordinates 
        u, w = rotate(c.û)

        # stratification
        Bz = c.N^2*cos(θ) .+ differentiate(c.b, ẑ)

        # colors and labels
        label = string("Day ", Int64(round(c.t/secsInDay)))
        color = colors[i, :]

        # plot
        ax[1, 1].plot(Bz,       z/1000, c=color, label=label)
        ax[1, 2].plot(c.χ,      z/1000, c=color, label=label)
        ax[1, 2].axvline(c.U,           c=color, lw=1.0, ls="--")
        ax[2, 1].plot(u,        z/1000, c=color, label=label)
        ax[2, 2].plot(c.v̂,      z/1000, c=color, label=label)
        axins21.plot(u,         z/1000, c=color, label=label)
    end

    ax[1, 2].legend()

    savefig("profiles.png")
    println("profiles.png")
end
