using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
    -∂zz(u) + u = ∂x(f),
BC:
    • u = 0 at z = 0
    • u = 0 at z = -H
    # • ∫ zu dz = 0
"""
function solve_toyDG1D(g)
    cols = get_cols(g.p, g.t)
    # N = sum(length(unique(g.t[cols[i], :][:])) for i ∈ eachindex(cols))
    N = 2*g.np - 2

    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for i ∈ eachindex(cols), k ∈ cols[i]
    end
end

function toyDG1D(; order)
end

function get_cols(p, t)
    # number of columns is half the number of boundary edges
    emap, edges, bndix = all_edges(t)
    bnd_edges = edges[bndix, :]
    ncols = Int64(1/2*size(bnd_edges, 1))

    # bounds to each column
    x = range(-1, 1, length=ncols+1)

    # k = cols[i][j] = jᵗʰ element of iᵗʰ col
    cols = [Int64[] for i=1:ncols]

    for k ∈ axes(t, 1)
        x̄ = sum(p[t[k, :], 1])/3
        i = searchsorted(x, x̄).stop
        push!(cols[i], k)
    end

    return cols
end

# p = [ 0.0   0.0
#       0.1   0.0
#       0.0  -0.25
#       0.1  -0.25
#       0.0  -0.5
#       0.1  -0.5
#       0.0  -0.75
#       0.1  -0.75
#       0.0  -1.0
#       0.1  -1.1]
# t = [i + j - 1 for i=1:size(p, 1)-2, j=1:3]
# tplot(p, t)
# axis("equal")
# ylim(-1.2, 0.1)
# savefig("scratch/images/mesh.png")
# println("scratch/images/mesh.png")

g = FEGrid("meshes/valign2D/mesh0.h5", 1)
cols = get_cols(g.p, g.t)
fig, ax, im = tplot(g)
ax.axis("equal")
cycle = [1, 2, 3, 1]
for i ∈ eachindex(cols)
    color = "C$(i-1)"
    for k ∈ cols[i]
        ax.plot(g.p[g.t[k, cycle], 1], g.p[g.t[k, cycle], 2], c=color, lw=0.5)
    end
end
savefig("scratch/images/cols.png")
println("scratch/images/cols.png")
plt.close()

g = FEGrid("meshes/valign2D/mesh0.h5", 1)
solve_toyDG1D(g)