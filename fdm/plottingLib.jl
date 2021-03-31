################################################################################
# Functions useful for plotting
################################################################################

pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")

"""
    ax = ridgePlot(field, b, titleString, cbarLabel; ax, vext, cmap)

Create 2D plot of `field` with isopycnals given by the buoyancy perturbation `b`.
Set the title to `titleString` and colorbar label to `cbarLabel`. Return the axis 
handle `ax`.

Optional: set the vmin/vmax manually with vext.
"""
function ridgePlot(field, b, titleString, cbarLabel; ax=nothing, vext=nothing, cmap="RdBu_r")
    # km
    xx = x/1000
    zz = z/1000

    # full buoyancy for isopycnals
    B = N^2*z + b 

    if ax == nothing
        fig, ax = subplots(1)
    end

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
    img = ax.pcolormesh(xx, zz, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    cb = colorbar(img, ax=ax, label=cbarLabel, extend=extend)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0))

    # isopycnal contours
    nLevels = 20
    lowerLevel = N^2*minimum(z)
    upperLevel = 0
    levels = lowerLevel:(upperLevel - lowerLevel)/(nLevels - 1):upperLevel
    ax.contour(xx, zz, B, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)

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

Plot profiles of b, u, v, w from HDF5 snapshot files of buoyancy in the `datafiles` list
at ξ = ξ[iξ].
"""
function profilePlot(datafiles, iξ)
    # init plot
    fig, ax = subplots(2, 3, figsize=(10, 6.5/1.62))

    # insets
    axins21 = inset_locator.inset_axes(ax[2, 1], width="40%", height="40%")

    ax[1, 1].set_xlabel(L"$B_z$ (s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$z$ (m)")
    ax[1, 1].set_title("stratification")

    ax[1, 2].set_xlabel(L"$\chi$ (m$^2$ s$^{-1}$)")
    ax[1, 2].set_ylabel(L"$z$ (m)")
    ax[1, 2].set_title("streamfunction")

    ax[1, 3].set_xlabel(L"$b_x$ (s$^{-2}$)")
    ax[1, 3].set_ylabel(L"$z$ (m)")
    ax[1, 3].set_title("buoyancy gradient")

    ax[2, 1].set_xlabel(L"$u$ (m s$^{-1}$)")
    ax[2, 1].set_ylabel(L"$z$ (m)")
    ax[2, 1].set_title("cross-ridge velocity")

    ax[2, 2].set_xlabel(L"$v$ (m s$^{-1}$)")
    ax[2, 2].set_ylabel(L"$z$ (m)")
    ax[2, 2].set_title("along-ridge velocity")

    ax[2, 3].set_xlabel(L"$w$ (m s$^{-1}$)")
    ax[2, 3].set_ylabel(L"$z$ (m)")
    ax[2, 3].set_title("vertical velocity")

    subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end
    axins21.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # zoomed z
    #= for a in ax =#
    #=     a.set_ylim([z[iξ, 1], z[iξ, 1] + 2000]) =#
    #= end =#
    ax[2, 1].set_ylim([z[iξ, 1], z[iξ, 1] + 200])

    # plot data from `datafiles`
    for i=1:size(datafiles, 1)
        # load
        c = loadCheckpointTF(string(folder, "2dpg/Pr1/checkpoint1000.h5"))
        u, v, w = transformFromTF(c.uξ, c.uη, c.uσ)

        # gradient
        bx = xDerivativeTF(c.b)

        # stratification
        Bz = c.N^2 .+ zDerivativeTF(c.b)

        # colors and labels
        label = string("Day ", Int64(round(c.t/86400)))
        color = colors[i, :]

        # plot
        ax[1, 1].plot(Bz[iξ, :],    z[iξ, :], c=color)
        ax[1, 2].plot(c.chi[iξ, :], z[iξ, :], c=color)
        ax[1, 3].plot(bx[iξ, :],    z[iξ, :], c=color)
        ax[2, 1].plot(u[iξ, :],     z[iξ, :], c=color)
        ax[2, 2].plot(v[iξ, :],     z[iξ, :], c=color)
        ax[2, 3].plot(w[iξ, :],     z[iξ, :], c=color, label=label)
        axins21.plot(u[iξ, :],      z[iξ, :], c=color)
    end

    ax[2, 3].legend()

    savefig("profiles.png")
end

#= """ =#
#=     advectionPlot(datafiles, iξ) =#

#= Plot advection terms from HDF5 snapshot files of buoyancy in the `datafiles` list =#
#= at for the full 2D domain as well as just at ξ = ξ[iξ]. =#
#= """ =#
#= function advectionPlot(datafiles, iξ) =#
#=     inversionLHS = lu(getInversionLHS()) =#

#=     figP, axP = subplots(2, 3, figsize=(6.5, 6.5/1.8), sharey=true) =#
#=     colors = pl.cm.viridis(range(1, 0, length=5)) =#

#=     for i=1:size(datafiles, 1) =#
#=         # load =#
#=         b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, ξVariation, κ = loadCheckpointTF(datafiles[i]) =#
    
#=         adv1 = -uξ.*ξDerivativeTF(b) =#
#=         adv2 = -uσ.*σDerivativeTF(b) =#
#=         adv3 = -N^2*uξ.*Hx.(ξξ).*σσ =#
#=         adv4 = -N^2*uσ.*H.(ξξ) =#
#=         diff = σDerivativeTF(κ.*(N^2 .+ σDerivativeTF(b)./H.(ξξ)))./H.(ξξ) =#
#=         sum = adv1 + adv2 + adv3 + adv4 + diff =#
    
#=         vmax = maximum([maximum(abs.(adv1)) maximum(abs.(adv2)) maximum(abs.(adv3)) maximum(abs.(adv4)) maximum(abs.(diff))]) =#
    
#=         fig, ax = subplots(2, 3, figsize=(6.5, 6.5/1.8), sharey=true, sharex=true) =#
#=         img = ax[1, 1].pcolormesh(ξξ/1000, σσ, adv1, vmin=-vmax, vmax=vmax, cmap="RdBu_r") =#
#=         colorbar(img, ax=ax[1, 1], label=L"-u^\xi b_\xi") =#
#=         img = ax[1, 2].pcolormesh(ξξ/1000, σσ, adv2, vmin=-vmax, vmax=vmax, cmap="RdBu_r") =#
#=         colorbar(img, ax=ax[1, 2], label=L"-u^\sigma b_\sigma") =#
#=         img = ax[1, 3].pcolormesh(ξξ/1000, σσ, adv3, vmin=-vmax, vmax=vmax, cmap="RdBu_r") =#
#=         colorbar(img, ax=ax[1, 3], label=L"-N^2u^\xi H_x\sigma") =#
#=         img = ax[2, 1].pcolormesh(ξξ/1000, σσ, adv4, vmin=-vmax, vmax=vmax, cmap="RdBu_r") =#
#=         colorbar(img, ax=ax[2, 1], label=L"-N^2u^\sigma H") =#
#=         img = ax[2, 2].pcolormesh(ξξ/1000, σσ, diff, vmin=-vmax, vmax=vmax, cmap="RdBu_r") =#
#=         cb = colorbar(img, ax=ax[2, 2], label=L"H^{-1}[\kappa(N^2 + H^{-1}b_\sigma)]_\sigma") =#
#=         cb.ax.ticklabel_format(style="sci", scilimits=(-3, 3)) =#
#=         img = ax[2, 3].pcolormesh(ξξ/1000, σσ, sum, vmin=-vmax, vmax=vmax, cmap="RdBu_r") =#
#=         cb = colorbar(img, ax=ax[2, 3], label=L"b_t") =#
#=         cb.ax.ticklabel_format(style="sci", scilimits=(-3, 3)) =#
#=         ax[1, 1].set_ylabel(L"\sigma") =#
#=         ax[2, 1].set_ylabel(L"\sigma") =#
#=         ax[2, 1].set_xlabel(L"\xi") =#
#=         ax[2, 2].set_xlabel(L"\xi") =#
#=         ax[2, 3].set_xlabel(L"\xi") =#
#=         fig.tight_layout() =#
#=         fig.savefig(@sprintf("evol%d.png", t/86400)) =#
    
#=         axP[1, 1].plot(adv1[iξ, :], σ, c=colors[i, :], label=string("Day ", Int64(t/86400))) =#
#=         axP[1, 2].plot(adv2[iξ, :], σ, c=colors[i, :]) =#
#=         axP[1, 3].plot(adv3[iξ, :], σ, c=colors[i, :]) =#
#=         axP[2, 1].plot(adv4[iξ, :], σ, c=colors[i, :]) =#
#=         axP[2, 2].plot(diff[iξ, :], σ, c=colors[i, :]) =#
#=         axP[2, 3].plot(sum[iξ, :], σ, c=colors[i, :]) =#
#=         axP[2, 3].plot(adv3[iξ, :] + adv4[iξ, :] + diff[iξ, :], σ, c="k", ls=":") =#
#=         axP[2, 3].plot(adv3[iξ, :] + diff[iξ, :], σ, c=colors[i, :], ls="--") =#
#=         axP[1, 1].set_xlabel(L"-u^\xi b_\xi") =#
#=         axP[1, 2].set_xlabel(L"-u^\sigma b_\sigma") =#
#=         axP[1, 3].set_xlabel(L"-N^2u^\xi H_x\sigma") =#
#=         axP[2, 1].set_xlabel(L"-N^2u^\sigma H") =#
#=         axP[2, 2].set_xlabel(L"H^{-1}[\kappa(N^2 + H^{-1}b_\sigma)]_\sigma") =#
#=         axP[2, 3].set_xlabel(L"b_t") =#
#=     end =#
#=     axP[1, 1].set_ylabel(L"\sigma") =#
#=     axP[2, 1].set_ylabel(L"\sigma") =#
#=     axP[2, 2].ticklabel_format(style="sci", scilimits=(0, 0)) =#
#=     axP[2, 3].ticklabel_format(style="sci", scilimits=(0, 0)) =#
#=     axP[1, 1].legend() =#
#=     figP.tight_layout() =#
#=     figP.savefig("evolProfiles_zoom.png") =#
#=     for ax in axP =#
#=         ax.set_xlim([-3e-12, 3e-12]) =#
#=     end =#
#=     figP.savefig("evolProfiles.png") =#
#= end =#

"""
    plotCurrentState(t, chi, chiEkman, uξ, uη, uσ, b, iImg)

Plot the buoyancy and velocity state of the model at time `t` using label number `iImg`.
"""
function plotCurrentState(t, chi, chiEkman, uξ, uη, uσ, b, iImg)
    # convert to physical coordinates 
    u, v, w = transformFromTF(uξ, uη, uσ)

    # plots
    ridgePlot(chi, b, @sprintf("streamfunction at t = %4d days", t/86400), L"$\chi$ (m$^2$ s$^{-1}$)")
    savefig(@sprintf("chi%03d.png", iImg))
    close()

    ridgePlot(chiEkman, b, @sprintf("streamfunction theory at t = %4d days", t/86400), L"$\chi$ (m$^2$ s$^{-1}$)")
    savefig(@sprintf("chiEkman%03d.png", iImg))
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
