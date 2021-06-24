pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

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
    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[1, 1].set_title("stratification")

    ax[1, 2].set_xlabel(L"$\chi$ (m$^2$ s$^{-1}$)")
    ax[1, 2].set_ylabel(L"$z$ (km)")
    ax[1, 2].set_title("streamfunction")

    ax[2, 1].set_xlabel(L"$u$ (m s$^{-1}$)")
    ax[2, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_title("cross-ridge velocity")

    ax[2, 2].set_xlabel(L"$v$ (m s$^{-1}$)")
    ax[2, 2].set_ylabel(L"$z$ (km)")
    ax[2, 2].set_title("along-ridge velocity")

    subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end
    axins21.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)-1))

    # zoomed z
    ax[2, 1].set_ylim([z[1]/1e3, (z[1] + 2e2)/1e3])

    # plot data from `datafiles`
    for i=1:size(datafiles, 1)
        # load
        c = loadCheckpoint1DTCPG(datafiles[i])

        # convert to physical coordinates 
        u, w = rotate(c.û)

        # stratification
        Bz = c.N^2*cos(θ) .+ differentiate(c.b, ẑ)

        # colors and labels
        if c.t == -42
            # steady state
            label = "steady state"
            color = "k"
        else
            label = string(Int64(round(c.t/secsInYear)), " years")
            if i==1
                color = "k"
            else
                color = colors[i-1, :]
            end
        end

        # plot
        ax[1, 1].plot(Bz,       z/1e3, c=color, label=label)
        ax[1, 2].plot(c.χ,      z/1e3, c=color, label=label)
        ax[1, 2].axvline(c.U,          c=color, lw=1.0, ls="--")
        ax[2, 1].plot(u,        z/1e3, c=color, label=label)
        ax[2, 2].plot(c.v̂,      z/1e3, c=color, label=label)
        axins21.plot(u,         z/1e3, c=color, label=label)
    end

    ax[1, 2].legend()

    savefig("profiles.png")
    println("profiles.png")
end

"""
    profilePlotBL(datafilesFull, datafilesBL)

Compare profiles of b from HDF5 snapshot files of buoyancy in the `datafilesFull` and `datafilesBL` lists.
"""
function profilePlotBL(datafilesFull, datafilesBL)
    # init plot
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62))

    ax[1, 1].set_xlabel(L"buoyancy, $b$ (m s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$\hat z$ (km)")

    ax[1, 2].set_xlabel(L"stratification, $\partial_{\hat z} B$ (s$^{-2}$)")

    ax[2, 1].set_xlabel(L"BL buoyancy, $b$ (m s$^{-2}$)")
    ax[2, 1].set_ylabel(L"$\hat z$ (km)")

    ax[2, 2].set_xlabel(L"BL stratification, $\partial_{\hat z} B$ (s$^{-2}$)")

    subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6)

    c = loadCheckpoint1DTCPG(datafilesBL[1])
    ax[1, 1].annotate(string(L"\sigma =", @sprintf("%1.2e", c.Pr)),                   (0.5, 0.6), xycoords="axes fraction")
    ax[1, 1].annotate(string(L"S =",      @sprintf("%1.2e", c.N^2*tan(c.θ)^2/c.f^2)), (0.5, 0.5), xycoords="axes fraction")

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafilesFull, 1)-1))

    # limits
    ax[1, 1].set_ylim([-c.H/1e3, 0])
    ax[1, 2].set_ylim([-c.H/1e3, 0])
    ax[2, 1].set_ylim([-c.H/1e3, -c.H/1e3 + 0.5])
    ax[2, 2].set_ylim([-c.H/1e3, -c.H/1e3 + 0.5])
    ax[2, 2].set_xlim([0, c.N^2/1.5])

    # plot data
    for i=1:size(datafilesFull, 1)
        # load
        c = loadCheckpoint1DTCPG(datafilesFull[i])
        cBL = loadCheckpoint1DTCPG(datafilesBL[i])

        # stratification
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ)
        BzBL = cBL.N^2*cos(cBL.θ) .+ differentiate(cBL.b, cBL.ẑ)

        # colors and labels
        if c.t == -42
            # steady state
            label = "steady state"
            color = "k"
        else
            label = string(Int64(round(c.t/secsInYear)), " years")
            if i==1
                color = "k"
            else
                color = colors[i-1, :]
            end
        end

        # plot
        ax[1, 1].plot(c.b,   c.ẑ/1e3, c=color, label=label)
        ax[2, 1].plot(c.b,   c.ẑ/1e3, c=color, label=label)
        ax[1, 2].plot(Bz,    c.ẑ/1e3, c=color, label=label)
        ax[2, 2].plot(Bz,    c.ẑ/1e3, c=color, label=label)
        ax[2, 1].plot(cBL.b, cBL.ẑ/1e3, c="k", ls=":")
        ax[1, 1].plot(cBL.b, cBL.ẑ/1e3, c="k", ls=":")
        ax[2, 2].plot(BzBL,  cBL.ẑ/1e3, c="k", ls=":")
        ax[1, 2].plot(BzBL,  cBL.ẑ/1e3, c="k", ls=":")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL theory"]
    ax[1, 1].legend(custom_handles, custom_labels)
    ax[1, 2].legend()
    
    savefig("profilesBL.png")
    println("profilesBL.png")
end
