################################################################################
# Functions useful for plotting
################################################################################

pc = 1/6 # one pica is 1/6th of an inch

"""
    ax = ridge_plot(m, s, field, title, cbar_label; ax, vext, cmap)

Create 2D plot of `field` with isopycnals given by the buoyancy perturbation `b`
from the model state `s`. Set the title to `title` and colorbar label to 
`cbar_label`. Return the axis handle `ax`.

Optional: 
    - provide `ax`
    - set the vmin/vmax manually with `vext`
    - set different colormap `cmap`
    - set `style` as either "contour" or "pcolormesh"
    - set colorbar orientation with `cb_orientation`
    - set xlabel with `xlabel`
    - set colorbar pad with `pad`
"""
function ridge_plot(m::ModelSetup2DPG, s::ModelState2DPG, field::Array{Float64,2}, 
                title::AbstractString, cbar_label::AbstractString; 
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
            cb = colorbar(img, ax=ax, label=cbar_label, orientation=cb_orientation, extend=extend)
        else
            cb = colorbar(img, ax=ax, label=cbar_label, orientation=cb_orientation, extend=extend, pad=pad)
        end
    elseif style == "pcolormesh"
        img = ax.pcolormesh(xx, zz, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
        if pad === nothing
            cb = colorbar(img, ax=ax, label=cbar_label, extend=extend, orientation=cb_orientation)
        else
            cb = colorbar(img, ax=ax, label=cbar_label, extend=extend, orientation=cb_orientation, pad=pad)
        end
        cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    else
        error("Unkown style: ", style)
    end

    # isopycnal contours
    n_levels = 20
    iξ = argmax(m.H)
    lower_level = -trapz(m.N2[iξ, :], m.z[iξ, :])
    # upper_level = 0
    upper_level = lower_level/100
    levels = lower_level:(upper_level - lower_level)/(n_levels - 1):upper_level
    ax.contour(xx, zz, s.b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)

    # ridge shading
    ax.fill_between(xx[:, 1], zz[:, 1], minimum(zz), color="k", alpha=0.3, lw=0.0)

    # labels
    ax.set_title(title)
    if xlabel === nothing
        ax.set_xlabel(L"Horizontal coordinate $x$ (km)")
    else
        ax.set_xlabel(xlabel)
    end
    ax.set_ylabel(L"Vertical coordinate $z$ (km)")
    ax.set_xlim([m.ξ[1]/1e3, (m.ξ[end] + m.ξ[2] - m.ξ[1])/1e3])

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    
    return ax
end

"""
    profile_plot(datafiles, iξ)

Plot profiles of χ, v, b_z from HDF5 snapshot files of buoyancy in the `datafiles` list
at ξ = ξ[iξ].
"""
function profile_plot(setup_file, state_files, iξ)
    # ModelSetup 
    m = load_setup_2D(setup_file)

    # init plot
    fig, ax = subplots(1, 3, figsize=(27*pc, 12*pc), sharey=true)

    ax[1].set_xlabel(string("Streamfunction\n", L"$\chi$ (m$^2$ s$^{-1}$)"))
    ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")

    ax[2].set_xlabel(string("Along-slope flow\n", L"$u^\eta$ (m s$^{-1}$)"))

    ax[3].set_xlabel(string("Stratification\n", L"$\partial_z b$ (s$^{-2}$)"))

    subplots_adjust(bottom=0.3, top=0.90, left=0.1, right=0.95, wspace=0.2, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=true)
    end

    # lims
    ax[1].set_ylim([m.z[iξ, 1]/1e3, 0])

    # color map
    pl = pyimport("matplotlib.pylab")
    colors = pl.cm.viridis(range(1, 0, length=size(state_files, 1)-1))

    # plot data from `datafiles`
    for i=1:size(state_files, 1)
        # load
        s = load_state_2D(state_files[i])
        u, v, w = transform_from_TF(m, s)

        # stratification
        bz = ∂z(m, s.b)

        # colors and labels
        if s.i[1]*m.Δt/secs_in_year > 1
            label = string(Int64(round(s.i[1]*m.Δt/secs_in_year)), " years")
        else
            label = string(Int64(round((s.i[1] - 1)*m.Δt/secs_in_day)), " days")
        end
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

    savefig(string(out_folder, "profiles.png"))
    println(string(out_folder, "profiles.png"))
end

"""
    plot_state(m, s, i_img)

Make some ridge plots of the current model state using the label number `i_img`.
"""
function plot_state(m::ModelSetup2DPG, s::ModelState2DPG, i_img::Int64)
    # convert to physical coordinates 
    u, v, w = transform_from_TF(m, s)

    # plots
    ridge_plot(m, s, s.χ, @sprintf("t = %4d years", s.i[1]*m.Δt/secs_in_year), L"Streamfunction $\chi$ (m$^2$ s$^{-1}$)")
    savefig(@sprintf("%schi%03d.png", out_folder, i_img))
    plt.close()

    ridge_plot(m, s, s.b, @sprintf("t = %4d years", s.i[1]*m.Δt/secs_in_year), L"Buoyancy $b$ (m s$^{-2}$)"; style="pcolormesh")
    savefig(@sprintf("%sb%03d.png", out_folder, i_img))
    plt.close()

    ridge_plot(m, s, v, @sprintf("t = %4d years", s.i[1]*m.Δt/secs_in_year), L"Along-ridge flow $u^\eta$ (m s$^{-1}$)"; style="pcolormesh")
    savefig(@sprintf("%sv%03d.png", out_folder, i_img))
    plt.close()
end

"""
    fig, ax, im = tplot(p, t, u=nothing)
    fig, ax, im = tplot(u)

If `u` === nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` === solution vector: Plot filled contour color plot of solution `u`.
"""
function tplot(p, t, u=nothing; fig=nothing, ax=nothing, cmap="RdBu_r", vext=nothing, lw=0.2, edgecolors="k", contour=false)
    if fig === nothing
        fig, ax = subplots(1)
    end

    if u === nothing
        im = ax.tripcolor(p[:, 1], p[:, 2], t[:, 1:3] .- 1, 0*t[:, 1], cmap="Greys", edgecolors=edgecolors, linewidth=lw, rasterized=true)
    else
        if vext === nothing
            vmax = maximum(abs.(u))
        else
            vmax = vext
        end

        if size(u, 1) == size(t, 1)
            # `u` represents values on triangle faces
            shading = "flat"
        elseif size(u, 1) == size(p, 1)
            # `u` represents values on triangle vertices
            shading = "gouraud"
        end

        im = ax.tripcolor(p[:, 1], p[:, 2], t[:, 1:3] .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading=shading, rasterized=true)
        if contour
            levels = range(-vmax, vmax, length=6)
            ax.tricontour(p[:, 1], p[:, 2], t[:, 1:3] .- 1, u, colors="k", linewidths=0.5, linestyles="-", levels=levels)
        end

    end

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end
function tplot(g::FEGrid; kwargs...)
    return tplot(g.p, g.t; kwargs...)
end
function tplot(u::FEField; kwargs...)
    return tplot(u.g.p, u.g.t, u.values; kwargs...)
end