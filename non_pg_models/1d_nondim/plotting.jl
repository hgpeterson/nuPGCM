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
    fig, ax = subplots(1, 3, figsize=(6.5, 2), sharey=true)


    ax[1].set_xlabel(L"cross-slope flow, $\tilde{u}$")
    ax[1].set_ylabel(L"$\tilde{z}$")

    ax[2].set_xlabel(L"along-slope flow, $\tilde{v}$")

    ax[3].set_xlabel(L"stratification, $\partial_{\tilde z} \tilde b$")

    subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.2, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)-1))

    # zoomed z
    ax[1].set_ylim([0, 10])

    # plot data from `datafiles`
    for i ∈ eachindex(datafiles)
        # load
        c = loadCheckpoint1DTCNondim(datafiles[i])

        # stratification
        b̃z̃ = differentiate(c.b̃, c.z̃)

        # colors and labels
        label = string(L"$\tilde{t}/\tilde{\tau}_A$ = ", Int64(round(c.t̃/τ_A)))
        if i == 1
            color = "k"
        else
            color = colors[i-1, :]
        end

        # plot
        ax[1].plot(c.ũ,  c.z̃, c=color)
        ax[2].plot(c.ṽ,  c.z̃, c=color)
        ax[2].axvline(c.P̃x̃, lw=1.0, c=color, ls="--")
        ax[3].plot(b̃z̃,   c.z̃, c=color, label=label)
    end

    ax[3].legend()

    savefig(fname)
    println(fname)
    plt.close()
end
