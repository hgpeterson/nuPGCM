### utility function for some of the stokes/laplace/pg tests

"""
    ebot, etop = get_sides(g)
"""
function get_sides(g)
    # bottom
    ebot = g.e[abs.(g.p[g.e, 2]) .>= 1e-4]
    eleft = g.e[abs.(g.p[g.e, 1] .+ 1) .<= 1e-4]
    eright = g.e[abs.(g.p[g.e, 1] .- 1) .<= 1e-4]
    ebot = [eleft[1]; ebot; eright[1]]

    # top
    etop = g.e[abs.(g.p[g.e, 2]) .< 1e-4]
    deleteat!(etop, findall(x->x==eleft[1], etop))
    deleteat!(etop, findall(x->x==eright[1], etop))
    return ebot, etop
end

"""
    quickplot(g, u, clabel, ofile)
"""
function quickplot(gu, u, clabel, ofile)
    fig, ax, im = tplot(gu.p, gu.t, u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end

"""
    quickplot(gb, b, gu, u, clabel, ofile)
"""
function quickplot(gb, b, gu, u, clabel, ofile)
    fig, ax, im = tplot(gu.p, gu.t, u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.tricontour(gb.p[:, 1], gb.p[:, 2], gb.t[:, 1:3] .- 1, b, linewidths=0.5, colors="k", linestyles="-", alpha=0.3)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end

"""
    quickplot(x, H, gb, b, gu, u, clabel, ofile)
"""
function quickplot(x, H, gb, b, gu, u, clabel, ofile)
    fig, ax, im = tplot(gu.p, gu.t, u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.tricontour(gb.p[:, 1], gb.p[:, 2], gb.t[:, 1:3] .- 1, b,
                  linewidths=0.5, colors="k", linestyles="-", alpha=0.3)
    ax.fill_between(x, -maximum(H), -H, color="k", alpha=0.3, lw=0.0)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end

