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
        im = ax.tripcolor(p[:,1], p[:,2], t .- 1, 0*t[:,1], cmap="Set3", edgecolors="k", linewidth=0.5)
    else
        if vext === nothing
            vmax = maximum(abs.(u))
        else
            vmax = vext
        end
        im = ax.tripcolor(p[:,1], p[:,2], t .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading="gouraud")
    end

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end
