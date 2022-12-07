using nuPGCM
using Delaunay
using HDF5
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function vertical_align(hx, hz; savefile=nothing)
    nx = Int64(ceil(2/hx))
    x = range(-1, 1, length=nx)

    H = @. 1 - x^2

    p = [-1.0  0.0]
    e = [1]
    for i=2:nx-1
        n = size(p, 1)
        nz = Int64(ceil(H[i]/hz))
        if nz == 2
            nz += 1
        end
        # z = range(-H[i], 0, length=nz)
        z = @. -H[i]*(cos(π*(0:nz-1)/(nz-1)) + 1)/2  
        p = vcat(p, [x[i]*ones(nz) z])
        push!(e, n+1)
        push!(e, size(p, 1))
    end
    p = vcat(p, [1.0  0.0])
    push!(e, size(p, 1))

    mesh = delaunay(p)
    t = mesh.simplices

    println("np = ", size(p, 1))
    if size(p, 1) < 1000
        fig, ax, im = tplot(p, t)
        ax.plot(p[:, 1], p[:, 2], "o", ms=1)
        ax.plot(p[e, 1], p[e, 2], "o", ms=1)
        ax.axis("equal")
        savefig("mesh.png")
        println("mesh.png")
        plt.close()
    end

    if savefile !== nothing
        h5open(savefile, "w") do file
            write(file, "p", p)
            write(file, "t", t)
            write(file, "e", e)
        end
        println(savefile)
    end

    return p, t, e
end

# hs = [0.16, 0.08, 0.04, 0.02, 0.01, 0.005]
# for i in eachindex(hs)
#     p, t, e = vertical_align(hs[i], hs[i]; savefile="valign/mesh$(i-1).h5")
# end

# p, t, e = vertical_align(0.01, 0.01; savefile="mesh.h5")

println("Done.")