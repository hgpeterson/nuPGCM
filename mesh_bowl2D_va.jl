using Delaunay
using PyPlot
using HDF5

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function mesh(h)
    x = range(-1, 1, step=h)
    y = x.^2 .- 1
    p = [-1. 0.; 1. 0.]
    e = [1, 2]
    np = 2
    for i ∈ 2:length(x)-1
        n = Int64(ceil(abs(y[i])/h)) + 1
        # z = range(y[i], 0, length=n)
        z = -y[i]*(chebyshev_nodes(n) .- 1)/2
        e = [e; np+1]
        for j ∈ eachindex(z)
            p = [p; x[i] z[j]]
            np += 1
        end
        e = [e; np]
    end
    t = delaunay(p).simplices

    if np < 1000
        fig, ax = plt.subplots(1)
        ax.tripcolor(p[:, 1], p[:, 2], t .- 1, zeros(np), cmap="Greys", edgecolor="k", lw=0.5)
        ax.plot(p[e, 1], p[e, 2], "o", ms=1)
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"z")
        savefig("images/mesh.png")
        println("images/mesh.png")
        plt.close()
    else
        println("Mesh too large to plot")
    end

    return p, t, e
end

p, t, e = mesh(0.01)

h5open("mesh.h5", "w") do file
    write(file, "p", p)
    write(file, "t", t)
    write(file, "e", e)
end
println("mesh.h5")