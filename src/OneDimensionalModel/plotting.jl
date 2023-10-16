pc = 1/6 # a pica is 1/6th of an inch

"""
    profile_plot(setupFile, stateFiles)

Plot profiles of b, χ, u, and v from HDF5 snapshot files.
"""
function profile_plot(setup_file, state_files)
    # ModelSetup 
    m = load_setup_1D(setup_file)

    # init plot
    fig, ax = subplots(2, 2, figsize=(18*pc, 23*pc), sharey=true)

    # insets
    inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
    axins21 = inset_locator.inset_axes(ax[2, 1], width="40%", height="40%")

    ax[1, 1].set_xlabel(latexstring(L"Stratification $\partial_z b'$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$ (km)")

    ax[1, 2].set_xlabel(latexstring(L"Streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))

    ax[2, 1].set_xlabel(latexstring(L"Cross-slope velocity $u^x$", "\n", L"($\times 10^{-4}$ m s$^{-1}$)"))
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$ (km)")

    ax[2, 2].set_xlabel(latexstring(L"Along-slope velocity $u^y$", "\n", L"($\times 10^{-2}$ m s$^{-1}$)"))

    subplots_adjust(hspace=0.5, wspace=0.3)

    # color map
    pl = pyimport("matplotlib.pylab")
    if string(out_folder, "state-1.h5") in state_files
        colors = pl.cm.viridis(range(1, 0, length=size(state_files, 1)-2))
    else
        colors = pl.cm.viridis(range(1, 0, length=size(state_files, 1)-1))
    end

    # lims
    ax[1, 1].set_ylim([m.z[1]/1e3, 0])
    ax[2, 1].set_ylim([m.z[1]/1e3, 0])
    axins21.set_ylim([m.z[1]/1e3, (m.z[1] + 1e2)/1e3]) # zoomed

    # plot data from `state_files`
    for i ∈ eachindex(state_files) 
        if i == 1 continue end # don't plot init cond

        # load
        s = load_state_1D(state_files[i])

        # colors and labels
        if s.i[1] == -1
            # steady state
            label = "steady state"
            color = "k"
        else
            label = string(Int64(round(s.i[1]*m.Δt/secs_in_year)), " years")
            if string(out_folder, "state-1.h5") in state_files
                color = colors[i-2, :]
            else
                color = colors[i-1, :]
            end
        end

        if m.bl
            # compute bl correction
            z = @. m.H*(1 - cos(pi*(0:m.nz-1)/(m.nz-1)))/2
            χ, b = get_full_soln(m, s, z)

            # compute u, v, Bz
            u = differentiate(χ, z)
            δ, μ, S, q = get_BL_params(m)
            v = @. -m.f*s.χ[1]/q/m.ν[1] - tan(m.θ)/m.f*(s.b - s.b[1]) # this is on the m.z grid
            Bz = m.N2 .+ differentiate(b, z)

            # plot
            ax[1, 1].plot(1e6*Bz, (z .- z[end])/1e3, c=color, label=label)
            ax[1, 2].plot(1e3*χ,  (z .- z[end])/1e3, c=color, label=label)
            ax[2, 1].plot(1e4*u,  (z .- z[end])/1e3, c=color, label=label)
            ax[2, 2].plot(1e2*v,  m./1e3,            c=color, label=label)
            axins21.plot(1e4*u,   (z .- z[end])/1e3, c=color, label=label)
        else
            # compute stratification
            Bz = m.N2 .+ differentiate(s.b, m.z)

            # plot
            ax[1, 1].plot(1e6*Bz,  m.z/1e3, c=color, label=label)
            ax[1, 2].plot(1e3*s.χ, m.z/1e3, c=color, label=label)
            ax[2, 1].plot(1e4*s.u, m.z/1e3, c=color, label=label)
            ax[2, 2].plot(1e2*s.v, m.z/1e3, c=color, label=label)
            axins21.plot(1e4*s.u,  m.z/1e3, c=color, label=label)
        end
    end

    ax[1, 2].legend()

    savefig(string(out_folder, "profiles.png"))
    println(string(out_folder, "profiles.png"))
    plt.close()
end