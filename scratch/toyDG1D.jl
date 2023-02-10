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
    -∂zz(u) + u = ∂x(b),
with
    • u = 0 at z = 0,
    • u = 0 at z = -H.
"""
function solve_toyDG1D(g, b)
    # for finding edge connectivities
    emap, edges, bndix = all_edges(g.t)

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
        end

        # boundary terms
        w, ξ = quad_weights_points(2, 1)
        s1D = ShapeFunctions(1, 1)
        # is vertical edge on left or right?
        # sign_multiplier = ...
        for i_e=1:3 # local edge index
            if emap[k, i_e] ∉ bndix # leave boundary edges for dirichlet
                # local indices for local edge i_e
                edge = [i_e, mod1(i_e+1, 3)]

                # parametric line 
                p1 = g.p[g.t[k, edge[1]], :]
                p2 = g.p[g.t[k, edge[2]], :]
                p(t) = (p2 - p1)/2*t + (p2 + p1)/2

                # connectivity pair for this edge
                edge_pair = findall(I -> emap[I] == emap[k, i_e] && I != CartesianIndex(k, i_e), CartesianIndices(emap))[1]

                # do ∫_∂Ωᵢₑ bφᵢ dz for each i
                for i=1:2
                    f(t) = b(p(t)[1], p(t)[2])*φ(s1D, i, t)*norm(p2 - p1)/2
                    ∫f = dot(w, f.(ξ))
                    # simple flux closure: take average of the two
                    r[t_dg[k, edge[i]]] += ∫f/2 
                    r[t_dg[edge_pair[1], edge[mod1(i+1, 2)]]] += ∫f/2 
                end
            end
        end
        # -∫_Ω b∂x(φ) dxdz
        r[t_dg[k, :]] -= Cx*b.(p_dg[t_dg[k, :], 1], p_dg[t_dg[k, :], 2])
    end

    # dirichlet
    for i ∈ e_dg
        push!(A, (i, i, 1))
        r[i] = 0
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

g = FEGrid("meshes/valign2D/mesh0.h5", 1)
b(x, z) = x
solve_toyDG1D(g, b)