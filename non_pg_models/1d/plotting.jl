pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

"""
    profilePlot(setupFile, stateFiles)

Plot profiles of b, u, and v from HDF5 snapshot files of buoyancy in the `datafiles` list.
"""
function profilePlot(setupFile::String, stateFiles::Vector{String})
    # ModelSetup 
    m = loadSetup1D(setupFile)

    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 6.5/1.62/2))

    # insets
    ax[1].set_xlabel(latexstring(L"cross-slope velocity $u^x$", "\n", L"($\times 10^{-4}$ m s$^{-1}$)"))
    ax[1].set_ylabel(L"$z$ (km)")

    ax[2].set_xlabel(latexstring(L"along-slope velocity $u^y$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))
    ax[2].set_ylabel(L"$z$ (km)")

    ax[3].set_xlabel(latexstring(L"stratification $\partial_z b$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))
    ax[3].set_ylabel(L"$z$ (km)")

    subplots_adjust(hspace=0.5, wspace=0.4)

    # lims
    ax[1].set_ylim([m.z[1]/1e3, (m.z[1] + 200)/1e3]) # zoomed
    ax[2].set_ylim([m.z[1]/1e3, 0])
    ax[3].set_ylim([m.z[1]/1e3, 0])

    # color map
    if string(outFolder, "state-1.h5") in stateFiles
        colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)-2))
    else
        colors = pl.cm.viridis(range(1, 0, length=size(stateFiles, 1)-1))
    end

    # plot data from `stateFiles`
    for i=1:size(stateFiles, 1)
        # load
        s = loadState1D(stateFiles[i])

        # stratification
        Bz = m.N2 .+ differentiate(s.b, m.z)

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

        # plot
        ax[1].plot(1e4*s.u, m.z/1e3, c=color, label=label)
        ax[2].plot(1e2*s.v, m.z/1e3, c=color, label=label)
        ax[2].axvline(1e2*s.∂ₓP[1]/m.f, c=color, ls="--", lw=1.0, label=label)
        ax[3].plot(1e6*Bz,  m.z/1e3, c=color, label=label)
    end

    ax[3].legend()

    savefig(string(outFolder, "profiles.png"))
    println(string(outFolder, "profiles.png"))
    plt.close()
end
function profilePlot(m::ModelSetup1D, u::Vector{Float64}, v::Vector{Float64}, 
    b::Vector{Float64}, ∂ₓP::Vector{Float64}, i::Vector{Int64}, imgFile::String)
    # init plot
    fig, ax = subplots(1, 3, figsize=(6.5, 6.5/1.62/2))

    ax[1].set_xlabel(latexstring(L"cross-slope velocity $u^x$", "\n", L"(m s$^{-1}$)"))
    ax[1].set_ylabel(L"$z$ (km)")

    ax[2].set_xlabel(latexstring(L"along-slope velocity $u^y$", "\n", L"(m s$^{-1}$)"))
    ax[2].set_ylabel(L"$z$ (km)")

    ax[3].set_xlabel(latexstring(L"stratification $\partial_z b$", "\n", L"(s$^{-2}$)"))
    ax[3].set_ylabel(L"$z$ (km)")

    subplots_adjust(hspace=0.5, wspace=0.4)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # lims
    ax[1].set_ylim([m.z[1]/1e3, (m.z[1] + 200)/1e3]) # zoomed
    ax[2].set_ylim([m.z[1]/1e3, 0])
    ax[3].set_ylim([m.z[1]/1e3, 0])

    ax[1].set_xlim([-1e-3, 1e-3])
    ax[2].set_xlim([-1e-3, 1e-3])
    ax[3].set_xlim([-1e-8, 1e-8])

    # stratification
    # Bz = m.N2 .+ differentiate(b, m.z)
    Bz = differentiate(b, m.z)

    # label
    label = string(Int64(round(i[1]*m.Δt/60/60)), " hours")

    # plot
    ax[1].plot(u, m.z/1e3, "k", label=label)
    ax[2].plot(v, m.z/1e3, "k", label=label)
    ax[2].axvline(∂ₓP[1]/m.f, color="k", ls="--", lw=1.0, label=label)
    ax[3].plot(Bz,  m.z/1e3, "k", label=label)
    
    ax[3].legend()

    savefig(imgFile)
    println(imgFile)
    plt.close()
end
function profilePlot(m::ModelSetup1D, stateFile::String, imgFile::String)
    s = loadState1D(stateFile)
    profilePlot(m, s.u, s.v, s.b, s.∂ₓP, s.i, imgFile)
end
function profilePlot(setupFile::String, stateFile::String, imgFile::String)
    m = loadSetup1D(setupFile)
    profilePlot(m, stateFile, imgFile)
end