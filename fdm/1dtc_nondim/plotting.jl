################################################################################
# Functions useful for plotting
################################################################################

pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")

"""
    profilePlot(datafiles)

Plot profiles from HDF5 snapshot files the `datafiles` list.
"""
function profilePlot(datafiles; fname="profiles.png")
    # init plot
    fig, ax = subplots(1, 3, figsize=(3.404*3, 3.404/1.62), sharey=true)

    #= ax[1].set_xlabel(L"$\tilde{B}_\tilde{z}$") =#
    ax[1].set_xlabel(L"$\partial_{\tilde z} \tilde b$")
    ax[1].set_ylabel(L"$\tilde{z}$")
    ax[1].set_title("stratification")

    ax[2].set_xlabel(L"$\tilde{u}$")
    ax[2].set_title("cross-ridge velocity")

    ax[3].set_xlabel(L"$\tilde{v}$")
    ax[3].set_title("along-ridge velocity")

    subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)-1))

    # zoomed z
    ax[1].set_ylim([0, 10])

    # plot data from `datafiles`
    for i=1:size(datafiles, 1)
        # load
        c = loadCheckpoint1DTCNondim(datafiles[i])

        # stratification
        #= Bz̃ = 1 .+ differentiate(c.b̃, c.z̃) =#
        b̃z̃ = differentiate(c.b̃, c.z̃)

        # colors and labels
        label = string(L"$\tilde{t}/\tilde{\tau}_A$ = ", Int64(round(c.t̃/τ_A)))
        if i == 1
            color = "k"
        else
            color = colors[i-1, :]
        end

        # plot
        #= ax[1].plot(Bz̃,   c.z̃, c=color, label=label) =#
        ax[1].plot(b̃z̃,   c.z̃, c=color, label=label)
        ax[2].plot(c.ũ,  c.z̃, c=color)
        ax[3].plot(c.ṽ,  c.z̃, c=color)
        ax[3].axvline(c.P̃x̃, lw=1.0, c=color, ls="--")
    end

    ax[1].legend()

    savefig(fname)
    println(fname)
    close()
end
