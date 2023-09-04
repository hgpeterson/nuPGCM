using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("images")

function plot_wireframe()
    H(x) = 1 - x[1]^2 - x[2]^2
    g_sfc1 = Grid(Triangle(order=1), "../meshes/circle/mesh1.h5")
    g_sfc2 = add_midpoints(g_sfc1)
    g1, g2, σ = nuPGCM.generate_wedge_cols(g_sfc1, g_sfc2)

    ax = plt.figure().add_subplot(projection="3d")
    edges = [[1, 2], [2, 3], [3, 1], 
             [1, 4], [2, 5], [3, 6], 
             [4, 5], [5, 6], [6, 4]] 
    for k=1:g1.nt
        for e ∈ edges
            if all(g1.p[g1.t[k, e], 2] .> 0)
                ax.plot(g1.p[g1.t[k, e], 1], g1.p[g1.t[k, e], 2], g1.p[g1.t[k, e], 3], "k-", lw=0.2)
            end
        end
    end
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.set_zlabel(L"z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 0)
    savefig("images/wireframe.png")
    println("images/wireframe.png")
end

plot_wireframe()