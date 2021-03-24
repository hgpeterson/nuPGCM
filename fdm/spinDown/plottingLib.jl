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

    ax[1].set_xlabel(L"$\tilde{B}_\tilde{z}$")
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
        c = loadCheckpointSpinDown(datafiles[i])

        # stratification
        Bẑ = 1 .+ differentiate(c.b, c.ẑ)

        # colors and labels
        label = string(L"$t/\tau_A$ = ", Int64(round(c.t/τ_A)))
        if i == 1
            color = "k"
        else
            color = colors[i-1, :]
        end

        # plot
        ax[1].plot(Bẑ,   c.ẑ, c=color, label=label)
        ax[2].plot(c.û,  c.ẑ, c=color)
        ax[3].plot(c.v,  c.ẑ, c=color)
        ax[3].axvline(c.Px, lw=1.0, c=color, ls="--")
    end

    ax[1].legend()

    savefig(fname)
    println(fname)
    close()
end

#= """ =#
#=     MR91plot(datafiles) =#

#= Plot profiles from HDF5 snapshot files the `datafiles` list as in MR91. =#
#= """ =#
#= function MR91plot(datafiles) =#
#=     # init plot =#
#=     fig, ax = subplots(1, 2, figsize=(3.404*2, 3.404/1.62), sharey=true) =#

#=     ax[1].set_xlabel(L"$\hat{u}, \hat{v}$ (cm s$^{-1}$)") =#
#=     ax[1].set_ylabel(L"$\hat{z}$ (cm)") =#
#=     ax[1].set_title("velocities") =#

#=     ax[2].set_xlabel(L"$\overline{\rho} + \rho'$ (g cm$^{-3}$)") =#
#=     ax[2].set_title("density") =#

#=     subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6) =#

#=     for a in ax =#
#=         a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0)) =#
#=     end =#

#=     # color map =#
#=     #1= colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)-1)) =1# =#
#=     colors = ["tab:blue", "tab:orange", "tab:green"] =#

#=     # xlim, ylim =#
#=     ax[1].set_xlim([-1.2, 0.4]) =#
#=     ax[1].set_ylim([0, 2]) =#
#=     ax[1].set_xticks(-1.2:0.2:0.4) =#
#=     ax[2].set_xlim([-0.01, 0.01]) =#

#=     # plot data from `datafiles` =#
#=     for i=1:size(datafiles, 1) =#
#=         # load =#
#=         û, v, b, Px, t, L, H0, Pr, f, N, θ, canonical, bottomIntense, κ, κ0, κ1, h, α = loadCheckpointSpinDown(datafiles[i]) =#

#=         # colors and labels =#
#=         #1= label = string("Day ", Int64(round(t/86400))) =1# =#
#=         label = string(Int64(round(t)), " s") =#
#=         c = colors[i] =#

#=         # plot =#
#=         ax[1].plot(100*û, 100*(ẑ .+ H0), c=c, ls="-") =#
#=         ax[1].plot(100*v, 100*(ẑ .+ H0), c=c, ls="--") =#
#=         ax[2].plot(-(b + N^2*(ẑ .+ H0))/9.81*1, 100*(ẑ .+ H0), c=c, label=label) =#
#=     end =#

#=     ax[2].legend() =#

#=     savefig("profiles.png", bbox="inches") =#
#=     close() =#
#= end =#
