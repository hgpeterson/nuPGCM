################################################################################
# Functions useful for plotting
################################################################################

# for colors
pl = pyimport("matplotlib.pylab")
mpl = pyimport("matplotlib")

"""
    ax = ridgePlot(m, s, field, titleString, cbarLabel; ax, vext, cmap)

Create 2D plot of `field` with isopycnals given by the buoyancy perturbation `b`
from the model state `s`. Set the title to `titleString` and colorbar label to 
`cbarLabel`. Return the axis handle `ax`.

Optional: 
    - provide `ax`
    - set the vmin/vmax manually with `vext`
    - set different colormap `cmap`
    - set `style` as either "contour" or "pcolormesh"
    - set colorbar orientation with `cb_orientation`
    - set xlabel with `xlabel`
    - set colorbar pad with `pad`
"""
function ridgePlot(m::ModelSetup2DPG, s::ModelState2DPG, field::Array{Float64,2}, 
                titleString::AbstractString, cbarLabel::AbstractString; 
                ax=nothing, vext=nothing, cmap="RdBu_r", style="contour",
                cb_orientation="vertical", xlabel=nothing, pad=nothing)
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
    else
        vmax = vext
        vmin = -vext
    end

    # set extend
    if maximum(field) > vmax && minimum(field) < vmin
        extend = "both"
    elseif maximum(field) > vmax && minimum(field) > vmin
        extend = "max"
    elseif maximum(field) < vmax && minimum(field) < vmin
        extend = "min"
    else
        extend = "neither"
    end


    # 2D plot
    if style == "contour"
        img = ax.pcolormesh(xx, zz, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
        levels = range(vmin, vmax, length=8)
        ax.contour(xx, zz, field, levels=levels, colors="k", linestyles="-", linewidths=0.25)
        if pad === nothing
            cb = colorbar(img, ax=ax, label=cbarLabel, orientation=cb_orientation, extend=extend)
        else
            cb = colorbar(img, ax=ax, label=cbarLabel, orientation=cb_orientation, extend=extend, pad=pad)
        end
    elseif style == "pcolormesh"
        img = ax.pcolormesh(xx, zz, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
        if pad === nothing
            cb = colorbar(img, ax=ax, label=cbarLabel, extend=extend, orientation=cb_orientation)
        else
            cb = colorbar(img, ax=ax, label=cbarLabel, extend=extend, orientation=cb_orientation, pad=pad)
        end
        cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    else
        error("Unkown style: ", style)
    end

    # isopycnal contours
    nLevels = 20
    lowerLevel = -trapz(m.N2[end, :], m.z[end, :])
    # upperLevel = 0
    upperLevel = lowerLevel/100
    levels = lowerLevel:(upperLevel - lowerLevel)/(nLevels - 1):upperLevel
    ax.contour(xx, zz, s.b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)

    # ridge shading
    ax.fill_between(xx[:, 1], zz[:, 1], minimum(zz), color="k", alpha=0.3, lw=0.0)

    # labels
    ax.set_title(titleString)
    if xlabel === nothing
        ax.set_xlabel(L"Horizontal coordinate $x$ (km)")
    else
        ax.set_xlabel(xlabel)
    end
    ax.set_ylabel(L"Vertical coordinate $z$ (km)")
    ax.set_xlim([0, m.L/1e3])

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    
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

    ax[1].set_xlabel(string("streamfunction\n", L"$\chi$ (m$^2$ s$^{-1}$)"))
    ax[1].set_ylabel(L"$z$ (km)")

    ax[2].set_xlabel(string("along-slope flow\n", L"$u^y$ (m s$^{-1}$)"))

    ax[3].set_xlabel(string("stratification\n", L"$\partial_z b$ (s$^{-2}$)"))

    subplots_adjust(bottom=0.3, top=0.90, left=0.1, right=0.95, wspace=0.2, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)-1))

    # plot data from `datafiles`
    for i=1:size(stateFiles, 1)
        # load
        s = loadState2DPG(stateFiles[i])
        u, v, w = transformFromTF(m, s)

        # stratification
        bz = zDerivative(m, s.b)

        # colors and labels
        label = string(Int64(round(s.i[1]*m.Δt/secsInYear)), " years")
        if i==1
            color = "tab:red"
        else
            color = colors[i-1, :]
        end

        # plot
        ax[1].plot(s.χ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
        ax[2].plot(v[iξ, :],   m.z[iξ, :]/1e3, c=color)
        ax[3].plot(bz[iξ, :],  m.z[iξ, :]/1e3, c=color)
    end

    ax[1].legend()

    savefig(string(outFolder, "profiles.png"))
    println(string(outFolder, "profiles.png"))
end

"""
    plotCurrentState(m, s, iImg)

Make some ridge plots of the current model state using the label number `iImg`.
"""
function plotCurrentState(m::ModelSetup2DPG, s::ModelState2DPG, iImg::Int64)
    # convert to physical coordinates 
    u, v, w = transformFromTF(m, s)

    # plots
    ridgePlot(m, s, s.χ, @sprintf("t = %4d years", s.i[1]*m.Δt/secsInYear), L"streamfunction $\chi$ (m$^2$ s$^{-1}$)")
    savefig(@sprintf("%schi%03d.png", outFolder, iImg))
    plt.close()

    ridgePlot(m, s, s.b, @sprintf("t = %4d years", s.i[1]*m.Δt/secsInYear), L"buoyancy $b$ (m s$^{-2}$)"; style="pcolormesh")
    savefig(@sprintf("%sb%03d.png", outFolder, iImg))
    plt.close()

    ridgePlot(m, s, v, @sprintf("t = %4d years", s.i[1]*m.Δt/secsInYear), L"along-ridge flow $v$ (m s$^{-1}$)"; style="pcolormesh")
    savefig(@sprintf("%sv%03d.png", outFolder, iImg))
    plt.close()
end

# function plot_advection(setupFile, stateFiles, iξ)
#     m = loadSetup2DPG(setupFile)

#     # total advection terms
#     fig, ax = subplots(1, 2, figsize=(2*1.955, 3.167), sharey=true)

#     ax[1].set_xlabel(string("Horizontal Advection", "\n", L"$u^\xi \partial_\xi b$ (m s$^{-3}$)"))
#     ax[2].set_xlabel(string("Vertical Advection", "\n", L"$u^\sigma \partial_\sigma b$ (m s$^{-3}$)"))
#     ax[1].set_ylabel(L"Vertical Coordinate $z$ (km)")
#     ax[2].annotate(string(L"$x = $", @sprintf("%d", round(m.ξ[iξ]/1e3, sigdigits=1)), " km"), (0.1, 0.8), xycoords="axes fraction")

#     ax[1].set_ylim([m.z[iξ, 1]/1e3, 0])
    
#     for a=ax
#         a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
#     end

#     # color map
#     colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)))

#     # plot data from `datafiles`
#     for i=1:size(stateFiles, 1)
#         # load
#         s = loadState2DPG(stateFiles[i])

#         # colors and labels
#         label = string(Int64(round(s.i[1]*m.Δt/secsInYear)), " years")
#         color = colors[i, :]

#         # gradients
#         dbdξ = ξDerivative(m, s.b)
#         dbdσ = σDerivative(m, s.b)

#         # plot
#         ax[1].plot(s.uξ[iξ, :].*dbdξ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
#         ax[2].plot(s.uσ[iξ, :].*dbdσ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
#     end

#     ax[1].legend()

#     savefig(string(outFolder, "advection_$iξ.png"))
#     println(string(outFolder, "advection_$iξ.png"))
#     plt.close()

#     # one term at a time
#     fig, ax = subplots(1, 2, figsize=(2*1.955, 3.167), sharey=true)

#     ax[1].set_xlabel(string("Horizontal Velocity", "\n", L"$u^\xi$ (m s$^{-1}$)"))
#     ax[2].set_xlabel(string("Horizontal Buoyancy Gradient", "\n", L"$\partial_\xi b$ (s$^{-2}$)"))
#     ax[1].set_ylabel(L"Vertical Coordinate $z$ (km)")
#     ax[2].annotate(string(L"$x = $", @sprintf("%d", round(m.ξ[iξ]/1e3, sigdigits=1)), " km"), (0.1, 0.8), xycoords="axes fraction")

#     ax[1].set_ylim([m.z[iξ, 1]/1e3, 0])
    
#     for a=ax
#         a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
#     end

#     # color map
#     colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)))

#     # plot data from `datafiles`
#     for i=1:size(stateFiles, 1)
#         # load
#         s = loadState2DPG(stateFiles[i])

#         # colors and labels
#         label = string(Int64(round(s.i[1]*m.Δt/secsInYear)), " years")
#         color = colors[i, :]

#         # gradients
#         dbdξ = ξDerivative(m, s.b)

#         # plot
#         ax[1].plot(s.uξ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
#         ax[2].plot(dbdξ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
#     end

#     ax[1].legend()

#     savefig(string(outFolder, "horizontal_$iξ.png"))
#     println(string(outFolder, "horizontal_$iξ.png"))
#     plt.close()

#     # one term at a time
#     fig, ax = subplots(1, 2, figsize=(2*1.955, 3.167), sharey=true)

#     ax[1].set_xlabel(string("Vertical Velocity", "\n", L"$u^\sigma$ (s$^{-1}$)"))
#     ax[2].set_xlabel(string("Vertical Buoyancy Gradient", "\n", L"$\partial_\sigma b$ (m s$^{-2}$)"))
#     ax[1].set_ylabel(L"Vertical Coordinate $z$ (km)")
#     ax[2].annotate(string(L"$x = $", @sprintf("%d", round(m.ξ[iξ]/1e3, sigdigits=1)), " km"), (0.1, 0.8), xycoords="axes fraction")

#     ax[1].set_ylim([m.z[iξ, 1]/1e3, 0])
    
#     for a=ax
#         a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
#     end

#     # color map
#     colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)))

#     # plot data from `datafiles`
#     for i=1:size(stateFiles, 1)
#         # load
#         s = loadState2DPG(stateFiles[i])

#         # colors and labels
#         label = string(Int64(round(s.i[1]*m.Δt/secsInYear)), " years")
#         color = colors[i, :]

#         # gradients
#         dbdσ = σDerivative(m, s.b)

#         # plot
#         ax[1].plot(s.uσ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
#         ax[2].plot(dbdσ[iξ, :], m.z[iξ, :]/1e3, c=color, label=label)
#     end

#     ax[1].legend()

#     savefig(string(outFolder, "vertical_$iξ.png"))
#     println(string(outFolder, "vertical_$iξ.png"))
#     plt.close()
# end