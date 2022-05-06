"""
    fig, ax, im = tplot(p, t, u=nothing)

If `u` === nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` === solution vector: Plot filled contour color plot of solution `u`.
"""
function tplot(p, t, u=nothing; ax=nothing, cmap="RdBu_r", vext=nothing)
    if ax === nothing
        fig, ax = subplots(1)
    end

    if u === nothing
        im = ax.tripcolor(p[:, 1], p[:, 2], t .- 1, 0*t[:, 1], cmap="Set3", edgecolors="k", linewidth=0.5)
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

        im = ax.tripcolor(p[:, 1], p[:, 2], t .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading=shading)
    end

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end

function plot_horizontal(p, t, u; vext=nothing, clabel="")
    if vext === nothing
        vext = maximum(abs.(u))
        extend = "neither"
    else
        extend = "both"
    end
    p = p/1e3 # km
    fig, ax, im = tplot(p, t, u; vext=vext)
    cb = colorbar(im, ax=ax, label=clabel, extend=extend)
    n = 6
    levels = vext*[collect(-(n-1)/n:1/n:-1/n)' collect(1/n:1/n:(n-1)/n)']
    ax.tricontour(p[:, 1], p[:, 2], t .- 1, u, linewidths=0.25, colors="k", linestyles="-", levels=levels)
    ax.set_xlabel(L"Horizontal coordinate $\xi$ (km)")
    ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
    ax.set_yticks(-5000:2500:5000)
    ax.axis("equal")
end
