using nuPGCM
using PyPlot
using PyCall
using LinearAlgebra

mlines = pyimport("matplotlib.lines")
Line2D = mlines.Line2D

plt.style.use("plots.mplstyle")
pygui(false)
plt.close()

function bottom_elements(ref)
    g = Grid(1, "meshes/circle/mesh$ref.h5")
    # emap, edges, bndix = all_edges(g.t)
    # h = 1/size(edges, 1)*sum(norm(g.p[edges[i, 1], :] - g.p[edges[i, 2], :]) for i in axes(edges, 1))
    h = 0.1
    println(h)
    H = FEField(x->1 - x[1]^2 - x[2]^2, g)
    nzs = [Int64(round(H[i]/h)) + 1 for i=1:g.np]
    fig, ax = tplot(g)
    ax.axis("equal")
    for k=1:g.nt
        nzs_k = nzs[g.t[k, :]] .- minimum(nzs[g.t[k, :]])
        if maximum(nzs_k) == 0
            ax.add_patch(plt.Polygon(g.p[g.t[k, :], :], lw=0, color="C0"))
        elseif maximum(nzs_k) == 1
            if length(findall(nzs_k .== 1)) == 1
                ax.add_patch(plt.Polygon(g.p[g.t[k, :], :], lw=0, color="C1"))
            else
                ax.add_patch(plt.Polygon(g.p[g.t[k, :], :], lw=0, color="C2"))
            end
        else
            ax.add_patch(plt.Polygon(g.p[g.t[k, :], :], lw=0, color="C3"))
        end
    end
    ax.set_title(L"H = 1 - x^2 - y^2")
    custom_handles = [Line2D([0], [0], ls="", marker="^", color="C0", lw=0),
                      Line2D([0], [0], ls="", marker="^", color="C1", lw=0),
                      Line2D([0], [0], ls="", marker="^", color="C2", lw=0),
                      Line2D([0], [0], ls="", marker="^", color="C3", lw=0)]
    custom_labels = ["All wedges", "One tetra", "One pyramid", "More than one needed"]
    ax.legend(custom_handles, custom_labels, loc=(0.9, 0.5))
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:1)
    savefig("scratch/images/bottom_elements.png")
    println("scratch/images/bottom_elements.png")
    plt.close()
end

bottom_elements(3)