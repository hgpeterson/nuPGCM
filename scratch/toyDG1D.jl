using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
    -∂z(v) + u = f,
    -∂z(u) + v = 0
BC:
    • u = 0 at z = 0
    • ∫ zu dz = 0
"""
function solve_toyDG1D()
    np = size(p, 1)
    nt = size(t, 1)
    nn = 1
    umap = 1:nn*nt
    vmap = nn*nt .+ umap
    N = 2*nn*nt

    for k=1:nt
        xk = p[t[k, :], 1]
        zk = p[t[k, :], 2]
        # φ(i, x, z) = (z - zk[mod1(i + 1, size(zk, 1))])/(zk[i] - zk[mod1(i + 1, size(zk, 1))])
        # φx(i, x, z) = 0
        # φz(i, x, z) = 1/(zk[i] - zk[mod1(i + 1, size(zk, 1))])
        φ(x, z) = 1
        φx(x, z) = 0
        φz(x, z) = 0
    end
end

function toyDG1D(; order)
end

p = [ 0.0   0.0
      0.1   0.0
      0.0  -0.25
      0.1  -0.25
      0.0  -0.5
      0.1  -0.5
      0.0  -0.75
      0.1  -0.75
      0.0  -1.0
      0.1  -1.1]
t = [i + j - 1 for i=1:size(p, 1)-2, j=1:3]

tplot(p, t)
axis("equal")
ylim(-1.2, 0.1)
savefig("scratch/images/mesh.png")
println("scratch/images/mesh.png")