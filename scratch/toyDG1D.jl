using nuPGCM
using WriteVTK
using PyPlot
using SparseArrays
using LinearAlgebra
using PyCall
using Printf

Polygon = pyimport("matplotlib.patches").Polygon

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
Solve
    -∂zz(u) + u = ∂x(b),
with
    • u = 0 at z = 0,
    • u = 0 at z = -H.
"""
function solve_toyDG1D()
    # for finding edge connectivities
    emap, edges, bndix = all_edges(g1.t)

    # for element matricies
    s = ShapeFunctionIntegrals(g.s, g.s)
    J = Jacobians(g1)

    # separate mesh into columns
    # cols = get_cols(g.p, g.t)

    # get p, t, e for dg mesh
    # p_dg, t_dg, e_dg = get_pte_dg(g.p, g.t, g.e, cols)
    p_dg = g.p
    t_dg = g.t
    e_dg = g.e
    x = p_dg[:, 1]
    z = p_dg[:, 2]
    N = size(p_dg, 1) # should be 2*g.np - 2

    # for now, b is just b(x, z) on the dg mesh
    b_dg = b.(x, z)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)

    # debug
    Kg = zeros(N, N)
    Mg = zeros(N, N)

    # for col ∈ cols, k ∈ col
    for k=1:g.nt
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        Cx = J.dets[k]*sum(s.CT.*J.Js[k, :, 1], dims=1)[1, :, :]
        M = J.dets[k]*s.M

        # interior terms
        for i=1:g.nn, j=1:g.nn
            if t_dg[k, i] ∉ e_dg
                # ∫_Ω ∂z(φᵢ)∂z(φⱼ) dxdz
                push!(A, (t_dg[k, i], t_dg[k, j], K[i, j]))
                # ∫_Ω φᵢφⱼ dxdz
                push!(A, (t_dg[k, i], t_dg[k, j], M[i, j]))
            end
            Kg[t_dg[k, i], t_dg[k, j]] += K[i, j]
            Mg[t_dg[k, i], t_dg[k, j]] += M[i, j]
        end

        # 1D quadrature
        w, ξ = quad_weights_points(deg=2, dim=1)

        # 1D shape functions
        s1D = ShapeFunctions(order=1, dim=1)

        # which edge is vertical edge
        ie, edge = vert_edge(g.p[g.t[k, 1:3], :])

        # is it on left or right? +1 if right, -1 if left
        sign_multiplier = side_of_vert_edge(g.p[g.t[k, 1:3], :], ie)

        # z-coords of edge nodes
        z1 = g.p[g.t[k, edge[1]], 2]
        z2 = g.p[g.t[k, edge[2]], 2]

        # # connectivity pair for this edge
        # pair = findall(I -> emap[I] == emap[k, ie] && I != CartesianIndex(k, ie), CartesianIndices(emap))[1]
        # k_pair = pair[1]
        # ie_pair = pair[2]
        # edge_pair = [ie_pair, mod1(ie_pair+1, 3)]
        # if g.p[g.t[k_pair, edge_pair]] != g.p[g.t[k, edge]]
        #     edge_pair = [mod1(ie_pair+1, 3), ie_pair]
        # end

        # average b values
        # b1 = (b_dg[t_dg[k, edge[1]]] + b_dg[t_dg[k_pair, edge_pair[1]]])/2
        # b2 = (b_dg[t_dg[k, edge[2]]] + b_dg[t_dg[k_pair, edge_pair[2]]])/2
        b1 = b_dg[t_dg[k, edge[1]]]
        b2 = b_dg[t_dg[k, edge[2]]]

        # ∫_∂Ωᵢₑ bφᵢ dz 
        for i=1:2
            f(t) = (b1*φ(s1D, 1, t) + b2*φ(s1D, 2, t))*φ(s1D, i, t)
            ∫f = dot(w, f.(ξ))*abs(z2 - z1)/2
            r[t_dg[k, edge[i]]] += sign_multiplier*∫f
        end

        # -∫_Ω b∂x(φ) dxdz
        r[t_dg[k, :]] -= Cx*b_dg[t_dg[k, :]]
    end

    # dirichlet
    for i ∈ e_dg
        push!(A, (i, i, 1))
        r[i] = 0
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # debug
    display(Kg)
    display(Mg)

    # solve
    u = A\r

    # exact solution
    u_a = @. -(bx(x, z)*exp(-z)*(-1 + exp(z))*(-1 + exp(H(x) + z)))/(1 + exp(H(x)))
    err = FEField(abs.(u - u_a), g, g1)
    println(@sprintf("Max error: %1.1e", maximum(abs.(u - u_a))))
    println(@sprintf("L2 error: %1.1e", L2norm(err, s, J)))

    # save as .vtu
    points = p_dg'
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, t_dg[i, :]) for i ∈ axes(t_dg, 1)]
    # cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TRIANGLE, t_dg[i, :]) for i ∈ axes(t_dg, 1)]
    vtk_grid("output/u.vtu", points, cells) do vtk
        vtk["u"] = u
        vtk["uₐ"] = u_a
    end
end

"""
    ie, edge = function vert_edge(p)

Given a triangle with points `p` of the form 
    <| or |>, 
find the local edge index and edge nodes of the vertical edge.
"""
function vert_edge(p)
    for ie=1:3
        edge = [ie, mod1(ie+1, 3)]
        if p[edge[1], 1] == p[edge[2], 1]
            return ie, edge
        end
    end
end

"""
Given a triangle with points `p` and a vertical edge at local edge index `ie`,
find which side the vertical edge is on. Return -1 for left, +1 for right.
"""
function side_of_vert_edge(p, ie)
    if p[mod1(ie+2, 3), 1] > p[ie, 1]
        return -1
    else
        return +1
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
    # new dg mesh
    p_dg = zeros(Float64, (2*size(p, 1)-2, 2))
    t_dg = zeros(Int64, size(t))
    e_dg = zeros(Int64, (2*size(e, 1)-2,))

    # current node index
    i_p = 0

    # current edge node index
    i_e = 0

    # all the nodes within each column will have a unique tag
    for col ∈ cols
        # new nodes in column
        t_col = t[col, :]
        nodes = sort(unique(t_col))
        n = size(nodes, 1)

        # add them to p_dg
        p_dg[i_p+1:i_p+n, :] = p[nodes, :]

        # mapping for new global node indices for each element
        tmap(i) = i_p + searchsorted(nodes, i).start
        for k ∈ col
            t_dg[k, :] = tmap.(t[k, :])
        end

        # add nodes that were on the edge to e_dg
        edge_nodes = i_p .+ findall(i -> nodes[i] ∈ e, 1:size(nodes, 1))
        e_dg[i_e+1:i_e+size(edge_nodes,1)] = edge_nodes

        # add to current node indices
        i_p += n
        i_e += size(edge_nodes, 1)
    end
    return p_dg, t_dg, e_dg
end

# g = FEGrid("meshes/valign2D/mesh0.h5", 1)
# fig, ax, im = tplot(g)
# for k ∈ axes(g.t, 1)
#     ie, edge = vert_edge(g.p[g.t[k, 1:3], :])
#     side = side_of_vert_edge(g.p[g.t[k, 1:3], :], ie)
#     ax.add_patch(Polygon(g.p[g.t[k, 1:3], :], facecolor = (side == 1 ? "tab:blue" : "tab:orange")))
# end
# ax.axis("equal")
# savefig("scratch/images/side_test.png")
# println("scratch/images/side_test.png")

# g = FEGrid("meshes/valign2D/mesh3.h5", 1)

# h = 0.05
# nz = Int64(round(H(0)/h)) + 1
# p = zeros(2*nz, 2)
# for i=1:nz
#     p[2i-1, :] = [0  -(i-1)*h]
#     p[2i,   :] = [h  -(i-1)*h]
# end

nz = 40
h = 2/(2nz - 3)
println(h)
p = zeros(2*nz, 2)
p[1, :] = [0 0]
p[2, :] = [h 0]
for i=2:nz-1
    p[2i-1, :] = [0  -(2i-3)*h/2]
    p[2i,   :] = [h  -(i-1)*h]
end
p[2nz-1, :] = [0 -(2nz-3)*h/2]
p[2nz,   :] = [h -(2nz-3)*h/2]

t = [i + j for i=1:2nz-2, j=0:2]
e = [1, 2, 2nz-1, 2nz]

fig, ax = subplots(1, figsize=(1, 3))
tplot(p, t, fig=fig, ax=ax)
ax.axis("equal")
ax.set_ylim(-1.1, 0.1)
savefig("mesh.png")
println("mesh.png")
plt.close()

g1 = FEGrid(p, t, e, 1)
g = FEGrid(p, t, e, 1)

b(x, z) = x
bx(x, z) = 1
# H(x) = 1 - x^2
H(x) = 1
solve_toyDG1D()

# h e
# 0 7.7e-3
# 1 2.5e-3
# 2 1.0e-3
# 3 6.0e-4