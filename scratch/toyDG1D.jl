using nuPGCM
using WriteVTK
using PyPlot
using SparseArrays
using LinearAlgebra

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
Solve
    -∂zz(u) + u = 0,
with
    • u = 0 at z = 0,
    • u = 0 at z = -H.
"""
function solve_toyDG1D(g)
    # for element matricies
    s = ShapeFunctionIntegrals(g.s, g.s)
    J = Jacobians(g)

    # separate mesh into columns
    cols = get_cols(g.p, g.t)

    # get p, t, e for dg mesh
    p_dg, t_dg, e_dg = get_pte_dg(g.p, g.t, g.e, cols)
    N = size(p_dg, 1) # should be 2*g.np - 2

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for col ∈ cols, k ∈ col
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M
        for i=1:g.nn, j=1:g.nn
            if t_dg[k, i] ∉ e_dg
                push!(A, (t_dg[k, i], t_dg[k, j], K[i, j]))
                push!(A, (t_dg[k, i], t_dg[k, j], M[i, j]))
            end
        end
        # r[t_dg[k, :]] = M*ones(g.nn)
    end

    # dirichlet
    for i ∈ e_dg
        push!(A, (i, i, 1))
        r[i] = 1
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # solve
    u = A\r

    # save as .vtu
    points = p_dg'
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, t_dg[i, :]) for i ∈ axes(t_dg, 1)]
    vtk_grid("output/u.vtu", points, cells) do vtk
        vtk["u"] = u
    end
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
        # find which column centroid in x lives in
        x̄ = sum(p[t[k, :], 1])/3
        i = searchsorted(x, x̄).stop
        push!(cols[i], k)
    end

    return cols
end

function get_pte_dg(p, t, e, cols)
    p_dg = [0.0 0.0]
    t_dg = zeros(Int64, size(t))
    e_dg = [0]
    # all the nodes within each column will have a unique tag
    for col ∈ cols
        # current size of p_dg
        np = size(p_dg, 1) - 1

        # new nodes in column
        t_col = t[col, :]
        nodes = sort(unique(t_col))

        # add them to p_dg
        p_dg = [p_dg; p[nodes, :]]

        # mapping for new global node indices for each element
        tmap(i) = np + searchsorted(nodes, i).start
        for k ∈ col
            t_dg[k, :] = tmap.(t[k, :])
        end

        # add nodes that were on the edge to e_dg
        edge_nodes = np .+ findall(i -> nodes[i] ∈ e, 1:size(nodes, 1))
        e_dg = [e_dg; edge_nodes]
    end
    return p_dg[2:end, :], t_dg, e_dg[2:end]
end

# g = FEGrid("meshes/valign2D/mesh0.h5", 1)
# cols = get_cols(g.p, g.t)
# fig, ax, im = tplot(g)
# ax.axis("equal")
# cycle = [1, 2, 3, 1]
# for i ∈ eachindex(cols)
#     color = "C$(i-1)"
#     for k ∈ cols[i]
#         ax.plot(g.p[g.t[k, cycle], 1], g.p[g.t[k, cycle], 2], c=color, lw=0.5)
#     end
# end
# savefig("scratch/images/cols.png")
# println("scratch/images/cols.png")
# plt.close()

# g = FEGrid("meshes/valign2D/mesh0.h5", 1)
# cols = get_cols(g.p, g.t)
# p_dg, t_dg, e_dg = get_pte_dg(g.p, g.t, g.e, cols)
# fig, ax, im = tplot(g)
# ax.axis("equal")
# cycle = [1, 2, 3, 1]
# for i ∈ eachindex(cols)
#     color = "C$(i-1)"
#     for k ∈ cols[i]
#         ax.plot(p_dg[t_dg[k, cycle], 1], p_dg[t_dg[k, cycle], 2], c=color, lw=0.5)
#         savefig("scratch/images/debug.png")
#         sleep(0.1)
#     end
# end
# savefig("scratch/images/debug.png")
# println("scratch/images/debug.png")
# plt.close()

g = FEGrid("meshes/valign2D/mesh5.h5", 1)
solve_toyDG1D(g)