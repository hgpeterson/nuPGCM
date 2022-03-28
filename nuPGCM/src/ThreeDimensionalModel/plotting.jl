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
        im = ax.tripcolor(p[:, 1], p[:, 2], t .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading="gouraud")
    end

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end

function plot_Ψ(p, t, Ψ; vext=nothing)
    p = p/1e3 # km
    Ψ = Ψ/1e9 # Sv
    fig, ax, im = tplot(p, t, Ψ; vext=vext)
    cb = colorbar(im, ax=ax, label=L"Streamfunction $\Psi$ (Sv)")
    Ψmax = maximum(abs.(Ψ))
    n = 6
    levels = Ψmax*[collect(-(n-1)/n:1/n:-1/n)' collect(1/n:1/n:(n-1)/n)']
    ax.tricontour(p[:, 1], p[:, 2], t .- 1, Ψ, linewidths=0.25, colors="k", linestyles="-", levels=levels)
    ax.set_xlabel(L"Horizontal coordintate $\xi$ (km)")
    ax.set_ylabel(L"Horizontal coordintate $\eta$ (km)")
    ax.axis("equal")
    savefig("psi.png")
    plt.close()
end
