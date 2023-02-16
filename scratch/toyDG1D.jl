using nuPGCM
using WriteVTK
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

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
function get_LHS(col)
    # DOF in column
    N = col.np

    # for element matricies
    J = Jacobians(col)
    s = ShapeFunctionIntegrals(col.s, col.s)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    for k=1:col.nt
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # interior terms
        for i=1:col.nn, j=1:col.nn
            if col.t[k, i] ∉ col.e
                # ∫_Ω ∂z(φᵢ)∂z(φⱼ) dxdz
                push!(A, (col.t[k, i], col.t[k, j], K[i, j]))
                # ∫_Ω φᵢφⱼ dxdz
                push!(A, (col.t[k, i], col.t[k, j], M[i, j]))
            end
        end
    end

    ### boundary terms 

    # surface nodes -> just u = 0
    sfc = col.e[col.p[col.e, 2] .== 0.0]
    for i ∈ sfc
        println("node $i: u = 0")
        push!(A, (i, i, 1))
    end

    # bottom nodes -> integral condition
    bot = col.e[col.p[col.e, 2] .!= 0.0]

    # 1D quadrature weights and points
    w, ξ = quad_weights_points(deg=4, dim=1)

    # 1D shape functions
    s1D = ShapeFunctions(order=1, dim=1)

    # map [-1, 1] to [x1, x2]
    x1 = minimum(col.p[bot, 1])
    x2 = maximum(col.p[bot, 1])
    x(t) = (x2 - x1)*t/2 + (x1 + x2)/2

    for i ∈ eachindex(bot)
        println("node $(bot[i]): ∫ φ$i (∫ zu dz) dx = 0")
        for k=1:col.nt
            # triangle's vertices
            p_tri = col.p[col.t[k, :], :]

            # find vertical edge
            ie, edge = vert_edge(p_tri)
            side = side_of_vert_edge(p_tri, ie)

            # determine which node on vertical edge is on top/bot
            vedge_bot = edge[argmin(p_tri[edge, 2])]
            vedge_top = edge[argmax(p_tri[edge, 2])]

            # call the other node the "corner"
            corner = mod1(edge[2]+1, 3)
            
            # z1 and z2 as a function of x
            function zj(x, j)
                if j == 2
                    vedge = vedge_top
                elseif j == 1
                    vedge = vedge_bot
                else
                    error("Invalid tag $j.")
                end

                if side == 1
                    # <|
                    zj1 = p_tri[corner, 2]
                    zj2 = p_tri[vedge, 2]
                else
                    # |>
                    zj1 = p_tri[vedge, 2]
                    zj2 = p_tri[corner, 2]
                end

                return zj1*(x - x2)/(x1 - x2) + zj2*(x - x1)/(x2 - x1)
            end

            for j=1:col.nn
                function ∫zφⱼdz(x)
                    z1 = zj(x, 1)
                    z2 = zj(x, 2)
                    z(t) = (z2 - z1)*t/2 + (z2 + z1)/2
                    f(t) = z(t)*φ(col.s, j, transform_to_ref_el([x, z(t)], p_tri))#*J.dets[k]
                    return dot(w, f.(ξ))*abs(z2 - z1)/2
                end
                f(t) = ∫zφⱼdz(x(t))*φ(s1D, i, t)
                ∫f = dot(w, f.(ξ))*abs(x2 - x1)/2
                push!(A, (bot[i], col.t[k, j], ∫f))
            end
        end
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    return lu(A)
end

function get_RHS(col)
    # DOF in column
    N = col.np

    # for element matricies
    s = ShapeFunctionIntegrals(col.s, col.s)

    # stamp
    r = zeros(N)
    for k=1:col.nt
        # triangle nodes
        p_tri = col.p[col.t[k, 1:3], :]

        # matrices
        J = Jacobians(col)
        Cx = J.dets[k]*sum(s.CT.*J.Js[k, :, 1], dims=1)[1, :, :]

        # 1D quadrature
        w, ξ = quad_weights_points(deg=2, dim=1)

        # 1D shape functions
        s1D = ShapeFunctions(order=1, dim=1)

        # which edge is vertical edge
        ie, edge = vert_edge(p_tri)

        # is it on left or right? +1 if right, -1 if left
        sign_multiplier = side_of_vert_edge(p_tri, ie)

        # z-coords of edge nodes
        z1 = p_tri[edge[1], 2]
        z2 = p_tri[edge[2], 2]

        # # connectivity pair for this edge
        # pair = connectivities[k, ie] # findall(I -> emap[I] == emap[k, ie] && I != CartesianIndex(k, ie), CartesianIndices(emap))[1]
        # k_pair = pair[1]
        # ie_pair = pair[2]
        # edge_pair = [ie_pair, mod1(ie_pair+1, 3)]
        # if g.p[g.t[k_pair, edge_pair]] != g.p[g.t[k, edge]]
        #     edge_pair = [mod1(ie_pair+1, 3), ie_pair]
        # end

        # average b values
        # b1 = (b_dg[t_dg[k, edge[1]]] + b_dg[t_dg[k_pair, edge_pair[1]]])/2
        # b2 = (b_dg[t_dg[k, edge[2]]] + b_dg[t_dg[k_pair, edge_pair[2]]])/2
        b1 = b(p_tri[edge[1], 1], p_tri[edge[1], 2])
        b2 = b(p_tri[edge[2], 1], p_tri[edge[2], 2])

        # ∫_∂Ωᵢₑ bφᵢ dz 
        for i=1:2
            f(t) = (b1*φ(s1D, 1, t) + b2*φ(s1D, 2, t))*φ(s1D, i, t)
            ∫f = dot(w, f.(ξ))*abs(z2 - z1)/2
            r[col.t[k, edge[i]]] += sign_multiplier*∫f
        end

        # -∫_Ω b∂x(φ) dxdz
        r[col.t[k, :]] -= Cx*b(p_tri[:, 1], p_tri[:, 2])
    end

    # dirichlet or integral
    for i ∈ col.e
        r[i] = 0
    end

    return r
end

function solve(col)
    # get LHSs
    A = get_LHS(col)

    # get RHSs
    r = get_RHS(col)

    # solve each column
    u = A\r

    # exact solution
    x = col.p[:, 1]
    z = col.p[:, 2]
    # u_a = @. -(bx(x, z)*exp(-z)*(-1 + exp(z))*(-1 + exp(H(x) + z)))/(1 + exp(H(x)))
    u_a = @. exp(-z)*(-1 + exp(z))*(bx(x, z) - (bx(x, z)*exp(H(x))*(-1 + exp(H(x)) - H(x)))/(-1 + exp(H(x)))^2 - (bx(x, z)*exp(H(x) + z)*(-1 + exp(H(x)) - H(x)))/(-1 + exp(H(x)))^2)
    err = FEField(abs.(u - u_a), col, col)
    println(@sprintf("Max error: %1.1e", maximum(abs.(u - u_a))))
    s = ShapeFunctionIntegrals(col.s, col.s)
    J = Jacobians(col)
    println(@sprintf("L2 error: %1.1e", L2norm(err, s, J)))

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, col.t[i, :]) for i ∈ axes(col.t, 1)]
    vtk_grid("output/u.vtu", col.p', cells) do vtk
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

nz = 20
h = 2/(2nz - 3)
println("h = ", h)
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
savefig("scratch/images/mesh.png")
println("scratch/images/mesh.png")
plt.close()

col = FEGrid(p, t, e, 1)

b(x, z) = x
bx(x, z) = 1
# H(x) = 1 - x^2
H(x) = 1
solve(col)