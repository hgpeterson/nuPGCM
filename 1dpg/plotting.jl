pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

"""
    profilePlot(setupFile, stateFiles)

Plot profiles of b, χ, u, and v from HDF5 snapshot files.
"""
function profilePlot(setupFile::String, stateFiles::Vector{String})
    # ModelSetup 
    m = loadSetup1DPG(setupFile)

    # init plot
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62))

    # insets
    axins21 = inset_locator.inset_axes(ax[2, 1], width="40%", height="40%")

    ax[1, 1].set_xlabel(latexstring(L"stratification $\partial_z b'$", "\n", L"(s$^{-2}$)"))
    ax[1, 1].set_ylabel(L"$z$ (km)")

    ax[1, 2].set_xlabel(latexstring(L"streamfunction $\chi$", "\n", L"(m$^2$ s$^{-1}$)"))
    ax[1, 2].set_ylabel(L"$z$ (km)")

    ax[2, 1].set_xlabel(latexstring(L"cross-slope velocity $u^x$", "\n", L"(m s$^{-1}$)"))
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 2].set_xlabel(latexstring(L"along-slope velocity $u^y$", "\n", L"(m s$^{-1}$)"))
    ax[2, 2].set_ylabel(L"$z$ (km)")

    subplots_adjust(hspace=0.5, wspace=0.3)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end
    axins21.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)

    # color map
    if string(outFolder, "state-1.h5") in stateFiles
        colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)-2))
    else
        colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)-1))
    end

    # lims
    ax[1, 1].set_ylim([m.z[1]/1e3, 0])
    ax[1, 2].set_ylim([m.z[1]/1e3, 0])
    axins21.set_ylim([m.z[1]/1e3, 0])
    ax[2, 1].set_ylim([m.z[1]/1e3, (m.z[1] + 1e2)/1e3]) # zoomed
    ax[2, 2].set_ylim([m.z[1]/1e3, 0])
    # ax[2, 2].set_ylim([m.z[1]/1e3, (m.z[1] + 1e2)/1e3]) # zoomed

    # plot data from `stateFiles`
    # for i=1:size(stateFiles, 1)
    for i=2:size(stateFiles, 1) # don't plot init cond
        # load
        s = loadState1DPG(stateFiles[i])

        # colors and labels
        if s.i[1] == -1
            # steady state
            label = "steady state"
            color = "k"
        else
            label = string(Int64(round(s.i[1]*m.Δt/secsInYear)), " years")
            if s.i[1] == 1
                color = "r"
            else
                if string(outFolder, "state-1.h5") in stateFiles
                    color = colors[i-2, :]
                else
                    color = colors[i-1, :]
                end
            end
        end

        if m.bl
            # compute bl correction
            z = @. m.H*(1 - cos(pi*(0:m.nz-1)/(m.nz-1)))/2
            χ, b = constructFullSolution(m, s, z)

            # compute u, v, Bz
            u = differentiate(χ, z)
            δ, μ, S, q = get_BL_params(m)
            v = @. -m.f*s.χ[1]/q/m.ν[1] - tan(m.θ)/m.f*(s.b - s.b[1])
            Bz = m.N2 .+ differentiate(b, z)

            # plot
            ax[1, 1].plot(Bz, (z .- z[end])/1e3, c=color, label=label)
            ax[1, 2].plot(χ,  (z .- z[end])/1e3, c=color, label=label)
            ax[2, 1].plot(u,  (z .- z[end])/1e3, c=color, label=label)
            ax[2, 2].plot(v,  (z .- z[end])/1e3, c=color, label=label)
            axins21.plot(u,   (z .- z[end])/1e3, c=color, label=label)
        else
            # compute stratification
            Bz = m.N2 .+ differentiate(s.b, m.z)

            # plot
            ax[1, 1].plot(Bz,  m.z/1e3, c=color, label=label)
            ax[1, 2].plot(s.χ, m.z/1e3, c=color, label=label)
            ax[2, 1].plot(s.u, m.z/1e3, c=color, label=label)
            ax[2, 2].plot(s.v, m.z/1e3, c=color, label=label)
            axins21.plot(s.u,  m.z/1e3, c=color, label=label)
        end
    end

    ax[1, 2].legend()

    savefig(string(outFolder, "profiles.png"))
    println(string(outFolder, "profiles.png"))
    plt.close()
end
function profilePlot(m::ModelSetup1DPG, stateFile::String, imgFile::String)
    # init plot
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62))

    # insets
    axins21 = inset_locator.inset_axes(ax[2, 1], width="40%", height="40%")

    ax[1, 1].set_xlabel(latexstring(L"stratification $\partial_z b$", "\n", L"(s$^{-2}$)"))
    ax[1, 1].set_ylabel(L"$z$ (km)")

    ax[1, 2].set_xlabel(latexstring(L"streamfunction $\chi$", "\n", L"(m$^2$ s$^{-1}$)"))
    ax[1, 2].set_ylabel(L"$z$ (km)")

    ax[2, 1].set_xlabel(latexstring(L"cross-slope velocity $u^x$", "\n", L"(m s$^{-1}$)"))
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 2].set_xlabel(latexstring(L"along-slope velocity $u^y$", "\n", L"(m s$^{-1}$)"))
    ax[2, 2].set_ylabel(L"$z$ (km)")

    subplots_adjust(hspace=0.5, wspace=0.3)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end
    axins21.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)

    # lims
    ax[1, 1].set_ylim([m.z[1]/1e3, 0])
    ax[1, 2].set_ylim([m.z[1]/1e3, 0])
    axins21.set_ylim([m.z[1]/1e3, 0])
    ax[2, 1].set_ylim([m.z[1]/1e3, (m.z[1] + 2e2)/1e3]) # zoomed
    ax[2, 2].set_ylim([m.z[1]/1e3, 0])

    # plot data from `stateFile`
    s = loadState1DPG(stateFile)

    # stratification
    Bz = m.N2 .+ differentiate(s.b, m.z)

    # label
    label = string(Int64(round((s.i[1] - 1)*m.Δt/(60*60))), " hours")

    # plot
    ax[1, 1].plot(Bz,  m.z/1e3, c="k", label=label)
    ax[1, 2].plot(s.χ, m.z/1e3, c="k", label=label)
    ax[2, 1].axvline(0, lw=1, c="k", ls="--", alpha=0.3)
    ax[2, 1].plot(s.u, m.z/1e3, c="k", label=label)
    ax[2, 2].plot(s.v, m.z/1e3, c="k", label=label)
    axins21.plot(s.u,  m.z/1e3, c="k", label=label)
    
    ax[1, 1].legend()

    savefig(imgFile)
    println(imgFile)
    plt.close()
end
function profilePlot(setupFile::String, stateFile::String, imgFile::String)
    m = loadSetup1DPG(setupFile)
    profilePlot(m, stateFile, imgFile)
end