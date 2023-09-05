using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("images")

function plot_wireframe()
    g_sfc1 = Grid(Triangle(order=1), "../meshes/circle/mesh1.h5")
    H = FEField(x->1 - x[1]^2 - x[2]^2, g_sfc1)
    g_sfc2 = add_midpoints(g_sfc1)
    g1, g2, σ = nuPGCM.generate_wedge_cols(g_sfc1, g_sfc2)
    nσ = length(σ)

    ax = plt.figure(figsize=(3.2, 3.2)).add_subplot(projection="3d")
    # ax.view_init(elev=30, azim=-90)
    edges = [[1, 2], [2, 3], [3, 1], 
             [1, 4], [2, 5], [3, 6], 
             [4, 5], [5, 6], [6, 4]] 
    for k=1:g1.nt
        mod(k-1, 100) == 0 ? println(k-1, "/", g1.nt) : nothing
        for e ∈ edges
            if all(g1.p[g1.t[k, e], 2] .> 0)
                if all(g1.p[g1.t[k, e], 3] .== 0)
                    alpha = 1.0
                elseif maximum(abs.(g1.p[g1.t[k, e], 2])) < 0.16
                    alpha = 1.0
                else
                    alpha = 0.15
                end
                ax.plot(g1.p[g1.t[k, e], 1], g1.p[g1.t[k, e], 2], g1.p[g1.t[k, e], 3].*H[nuPGCM.get_i_sfc.(g1.t[k, e], length(σ))], "k-", lw=0.05, alpha=alpha)
                # ax.plot(g1.p[g1.t[k, e], 1], g1.p[g1.t[k, e], 2], g1.p[g1.t[k, e], 3], "k-", lw=0.05, alpha=alpha)
            end
        end
    end
    ax.axis("off")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 0)
    # ax.annotate("", xy=(0.320, 0.015), xytext=(0.1, 0.1), xycoords="axes fraction", arrowprops=Dict("arrowstyle"=>"-|>", "fc"=>"k"))
    # ax.annotate("", xy=(0.250, 0.250), xytext=(0.1, 0.1), xycoords="axes fraction", arrowprops=Dict("arrowstyle"=>"-|>", "fc"=>"k"))
    # ax.annotate("", xy=(0.100, 0.350), xytext=(0.1, 0.1), xycoords="axes fraction", arrowprops=Dict("arrowstyle"=>"-|>", "fc"=>"k"))
    savefig("$out_folder/wireframe.png")
    println("$out_folder/wireframe.png")
    plt.close()
end

plot_wireframe()