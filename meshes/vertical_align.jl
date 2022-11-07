using nuPGCM
using Delaunay
using HDF5
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function vertical_align(hx, hz)
    nx = Int64(ceil(2/hx))
    x = range(-1, 1, length=nx)

    H = @. sqrt(2 - x^2) - 1

    p = [-1.0       0.0
         -1.0+hx    0.0
         -1.0+hx   -H[2]]
    e = [1, 2, 3]
    for i=3:nx-2
        n = size(p, 1)
        nz = Int64(ceil(H[i]/hz))
        if nz == 2
            nz += 1
        end
        # if rand() > 0.5
        #     nz += 1
        # end
        z = range(-H[i], 0, length=nz)
        p = vcat(p, [x[i]*ones(nz) z])
        push!(e, n+1)
        push!(e, size(p, 1))
    end
    n = size(p, 1)
    p = vcat(p, [1.0-hx    0.0
                 1.0-hx   -H[end-1]
                 1.0       0.0])
    push!(e, n+1, n+2, n+3)

    # p = [-1.0  0.0
    #       0.0  0.0
    #       0.0 -0.5
    #       0.0 -1.0
    #       1.0  0.0]
    # e = [1, 2, 4, 5]

    mesh = delaunay(p)
    t = mesh.simplices

    println("np = ", size(p, 1))
    if size(p, 1) < 10000
        fig, ax, im = tplot(p, t)
        ax.plot(p[:, 1], p[:, 2], ".", ms=1)
        ax.plot(p[e, 1], p[e, 2], ".", ms=1)
        ax.axis("equal")
        savefig("mesh.png")
        println("mesh.png")
        plt.close()
    end

    file = h5open("mesh0.h5", "w")
    write(file, "p", p)
    write(file, "t", t)
    write(file, "e", e)
    close(file)
    println("mesh0.h5")

    return p, t, e
end

p, t, e = vertical_align(0.02, 0.02)
println("Done.")