using nuPGCM
using WriteVTK
using Delaunay
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
Solve
    -ε²∂zz(ωˣ) - ωʸ = 0,
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
       ∂zz(χˣ) + ωˣ = 0,
       ∂zz(χʸ) + ωʸ = 0,
with bc
At z = 0:
    • ωˣ = 0, ωʸ = 0, χˣ = 0, χʸ = -Uˣ
At z = -H:
    • ωˣ = Uˣ/ε², χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0
"""
function get_LHS(; col_i)
    # column
    col = cols[col_i]

    # indices
    ωxmap = 1:col.np
    ωymap = (col.np+1):2*col.np
    χxmap = (2*col.np+1):3*col.np
    χymap = (3*col.np+1):4*col.np
    N = 4*col.np

    # sfc and bot
    sfc = col.e[col.p[col.e, 2] .== 0.0]
    bot = col.e[col.p[col.e, 2] .!= 0.0]

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
            # indices
            ωxi = ωxmap[col.t[k, :]]
            ωyi = ωymap[col.t[k, :]]
            χxi = χxmap[col.t[k, :]]
            χyi = χymap[col.t[k, :]]
            if col.t[k, i] ∉ col.e
                # eq 1: ε²∂z(ωˣ)∂z(ωˣ)
                push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
                # eq 1: -ωʸωˣ
                push!(A, (ωxi[i], ωyi[j], -M[i, j]))

                # eq 2: ε²∂z(ωʸ)∂z(ωʸ)
                push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
                # eq 2: ωˣωʸ
                push!(A, (ωyi[i], ωxi[j],  M[i, j]))
            end
            if col.t[k, i] ∉ sfc
                # eq 3: -∂z(χˣ)∂z(χˣ)
                push!(A, (χxi[i], χxi[j], -K[i, j]))
                # eq 3: ωˣχˣ
                push!(A, (χxi[i], ωxi[j],  M[i, j]))

                # eq 4: ∂z(χʸ)∂z(χʸ)
                push!(A, (χyi[i], χyi[j], -K[i, j]))
                # eq 4: ωʸχʸ
                push!(A, (χyi[i], ωyi[j],  M[i, j]))
            end
        end
    end

    # surface nodes 
    for i ∈ sfc
        push!(A, (ωxmap[i], ωxmap[i], 1))
        push!(A, (ωymap[i], ωymap[i], 1))
        push!(A, (χxmap[i], χxmap[i], 1))
        push!(A, (χymap[i], χymap[i], 1))
    end

    # bottom nodes
    for i ∈ bot
        push!(A, (ωxmap[i], ωxmap[i], 1))
        push!(A, (ωymap[i], χymap[i], 1))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    return lu(A)
end

function get_RHS(; col_i)
    # column
    col = cols[col_i]

    # indices
    ωxmap = 1:col.np
    ωymap = (col.np+1):2*col.np
    χxmap = (2*col.np+1):3*col.np
    χymap = (3*col.np+1):4*col.np
    N = 4*col.np

    # for surface integrals
    s = ShapeFunctionIntegrals(col.s, col.s)
    J = Jacobians(col)

    # for edge integrals
    w, ξ = quad_weights_points(deg=2*col.order, dim=1)
    s1D = ShapeFunctions(order=col.order, dim=1)

    # stamp
    r = zeros(N)
    for k=1:col.nt
        # triangle nodes
        p_tri = col.p[col.t[k, 1:3], :]

        # which edge is vertical edge
        ie, edge = vert_edge(p_tri)
        node1 = col.t[k, edge[1]]
        node2 = col.t[k, edge[2]]

        # is it on left or right? +1 if right, -1 if left
        sign_multiplier = side_of_vert_edge(p_tri, ie)

        # z-coords of edge nodes
        z1 = col.p[node1, 2]
        z2 = col.p[node2, 2]

        # connected nodes in neighboring column 
        col_j, conn_node1 = node_conns[col_i][node1]
        col_j, conn_node2 = node_conns[col_i][node2]

        # average b values        
        b1 = (b_cols[col_i][node1] + b_cols[col_j][conn_node1])/2
        b2 = (b_cols[col_i][node2] + b_cols[col_j][conn_node2])/2

        # ∫_∂Ωᵢₑ bφᵢ dz 
        for i=1:2
            f(t) = (b1*φ(s1D, 1, t) + b2*φ(s1D, 2, t))*φ(s1D, i, t)
            ∫f = dot(w, f.(ξ))*abs(z2 - z1)/2
            r[ωymap[col.t[k, edge[i]]]] -= sign_multiplier*∫f
        end

        # -∫_Ω b∂x(φ) dxdz
        Cx = J.dets[k]*sum(s.CT.*J.Js[k, :, 1], dims=1)[1, :, :]
        r[ωymap[col.t[k, :]]] += Cx*b_cols[col_i][col.t[k, :]]
    end

    # surface nodes 
    sfc = col.e[col.p[col.e, 2] .== 0.0]
    for i ∈ sfc
        r[ωxmap[i]] = 0
        r[ωymap[i]] = 0
        r[χxmap[i]] = 0
        r[χymap[i]] = -Ux
    end

    # bottom nodes
    bot = col.e[col.p[col.e, 2] .!= 0.0]
    for i ∈ bot
        r[ωxmap[i]] = Ux/ε²
        r[ωymap[i]] = 0
    end

    return r
end

function solve(; col_i)
    # column
    col = cols[col_i]

    # indices
    ωxmap = 1:col.np
    ωymap = (col.np+1):2*col.np
    χxmap = (2*col.np+1):3*col.np
    χymap = (3*col.np+1):4*col.np

    # get LHSs
    A = get_LHS(col_i=col_i)

    # get RHSs
    r = get_RHS(col_i=col_i)

    # solve each column
    sol = A\r
    ωx = FEField(sol[ωxmap], col, col)
    ωy = FEField(sol[ωymap], col, col)
    χx = FEField(sol[χxmap], col, col)
    χy = FEField(sol[χymap], col, col)

    # compare with high res FD solution
    x = col.p[1, 1]
    z = -H(x):H(x)/2^10:0
    ωx_fd, ωy_fd = fd_sol(z, bx.(x, z), ε², Ux)
    ωx_f(z) = evaluate(ωx, [x, z])
    ωy_f(z) = evaluate(ωy, [x, z])
    χx_f(z) = evaluate(χx, [x, z])
    χy_f(z) = evaluate(χy, [x, z])
    println(@sprintf("Max error ωx: %1.1e", maximum(abs.(ωx_f.(z) - ωx_fd))))
    println(@sprintf("Max error ωy: %1.1e", maximum(abs.(ωy_f.(z) - ωy_fd))))

    # plot
    fig, ax = subplots(1, 2, figsize=(2*2, 3.2), sharey=true)
    ax[1].plot(ωx_f.(z), z, label=L"\omega^x")
    ax[1].plot(ωy_f.(z), z, label=L"\omega^y")
    ax[1].plot(ωx_fd, z, "k--", lw=0.5, label="“Truth”")
    ax[1].plot(ωy_fd, z, "k--", lw=0.5)
    ax[2].plot(χx_f.(z), z, label=L"\chi^x")
    ax[2].plot(χy_f.(z), z, label=L"\chi^y")
    ax[1].legend()
    ax[2].legend()
    ax[1].set_xlabel(L"\omega")
    ax[1].set_ylabel(L"z")
    ax[2].set_xlabel(L"\chi")
    savefig("scratch/images/omega_chi.png")
    println("scratch/images/omega_chi.png")
    plt.close()
end

function fd_sol(z, bx, ε², Ux)
    # indices
    nz = size(z, 1)
    ωxmap = 1:nz
    ωymap = (nz+1):2*nz

    # matrix
    A = Tuple{Int64,Int64,Float64}[]  
    r = zeros(2*nz)

    # interior nodes
    for j=2:nz-1 
        # ∂zz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # eqtn 1: -ε²∂zz(ωˣ) - ωʸ = 0
        # term 1
        push!(A, (ωxmap[j], ωxmap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωxmap[j], ωxmap[j],   -ε²*fd_zz[2]))
        push!(A, (ωxmap[j], ωxmap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωxmap[j], ωymap[j], -1))

        # eqtn 2: -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
        # term 1
        push!(A, (ωymap[j], ωymap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωymap[j], ωymap[j],   -ε²*fd_zz[2]))
        push!(A, (ωymap[j], ωymap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωymap[j], ωxmap[j], 1))
        # rhs
        r[ωymap[j]] = -bx[j]
    end

    # ωˣ = ωʸ = 0 at z = 0
    push!(A, (ωxmap[nz], ωxmap[nz], 1))
    push!(A, (ωymap[nz], ωymap[nz], 1))

    # ωˣ = Uˣ/ε² at z = -H
    push!(A, (ωxmap[1], ωxmap[1], 1))
    r[ωxmap[1]] = Ux/ε²

    # ∫ zωʸ dz = -Ux
    for j=1:nz-1
        push!(A, (ωymap[1], ωymap[j],     z[j]*(z[j+1] - z[j])/2))
        push!(A, (ωymap[1], ωymap[j+1], z[j+1]*(z[j+1] - z[j])/2))
    end
    r[ωymap[1]] = -Ux

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), 2*nz, 2*nz)

    sol = A\r
    return sol[ωxmap], sol[ωymap]
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

function gen_mesh(h)
    # surface "mesh"
    nx = Int64(ceil(2/h))
    x = range(-1, 1, length=nx)

    # store an array of columns
    cols = Vector{FEGrid}(undef, nx-1)

    # store node connectivities: col_j, conn_node = node_conns[col_i][node]
    node_conns = [[] for i=1:nx-1]

    # mesh each column
    left = [-1.0 0.0]
    p = copy(left)
    left_e = [1]
    e = copy(left_e)
    t = [0 0 0]
    push!(node_conns[1], nothing) # left corner has no connections
    for i=2:nx
        # column index
        col_i = i - 1

        # "mesh" right side of column
        if i == nx
            right = [1.0 0.0]
            right_e = [1]
        else
            nz = Int64(ceil(H(x[i])/h)) 
            nz += 1*(nz == 2)
            z = range(-H(x[i]), 0, length=nz)
            right = [x[i]*ones(nz)  z]
            right_e = [1, nz]
        end

        # points, triangles, and edges of column
        p_col = [left; right]
        t_col = delaunay(p_col).simplices
        e_col = [left_e; size(left, 1) .+ right_e]
        cols[col_i] = FEGrid(p_col, t_col, e_col, 1)

        # save node connectivities
        if col_i < nx - 1
            for j ∈ axes(right, 1)
                push!(node_conns[col_i+1], [col_i, size(left, 1) + j])
                push!(node_conns[col_i], [col_i+1, j])
            end
        end

        # add to global mesh
        e = [e; size(p, 1) .+ right_e]
        t = [t; size(p, 1) .- size(left, 1) .+ t_col]
        p = [p; right]

        # right is new left
        left = right
        left_e = right_e
    end
    push!(node_conns[end], nothing) # right corner has no connections
    t = t[2:end, :]

    g = FEGrid(p, t, e, 1)

    println("np = ", g.np)
    println("ncol = ", nx-1)

    # # plot full mesh
    # fig, ax, im = tplot(p, t)
    # ax.plot(p[:, 1], p[:, 2], "o", ms=1)
    # ax.plot(p[e, 1], p[e, 2], "o", ms=1)
    # ax.axis("equal")
    # savefig("scratch/images/full_mesh.png")
    # println("scratch/images/full_mesh.png")
    # plt.close()
    
    # # plot cols
    # fig, ax, im = tplot(p, t)
    # ax.axis("equal")
    # for i ∈ eachindex(cols)
    #     tplot(cols[i], fig=fig, ax=ax, edgecolors="C$i")
    # end
    # savefig("scratch/images/cols.png")
    # println("scratch/images/cols.png")
    # plt.close()

    return g, cols, node_conns
end

function plot_2D()
    # global p, t, e
    np = 2*g.np-2
    nt = g.nt
    ne = 2*g.ne-2
    p = zeros(Float64, (np, 2))
    t = zeros(Int64, (nt, 3))
    e = zeros(Int64, (ne,))

    # global solutions
    ωx = zeros(np)
    ωy = zeros(np)
    χx = zeros(np)
    χy = zeros(np)

    # current indices
    i_p = 0
    i_t = 0
    i_e = 0

    # all the nodes within each column will have a unique tag
    for i ∈ eachindex(cols)
        # column
        col = cols[i]

        # add nodes, triangles, and edge nodes
        p[i_p+1:i_p+col.np, :] = col.p
        t[i_t+1:i_t+col.nt, :] = i_p .+ col.t
        e[i_e+1:i_e+col.ne] = i_e .+ col.e

        # unpack solutions
        ωx[i_p+1:i_p+col.np] = sols[i][0*col.np+1:1*col.np]
        ωy[i_p+1:i_p+col.np] = sols[i][1*col.np+1:2*col.np]
        χx[i_p+1:i_p+col.np] = sols[i][2*col.np+1:3*col.np]
        χy[i_p+1:i_p+col.np] = sols[i][3*col.np+1:4*col.np]

        # increment
        i_p += col.np
        i_t += col.nt
        i_e += col.ne
    end

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid("output/pg_vort_DG.vtu", p', cells) do vtk
        vtk["ωx"] = ωx
        vtk["ωy"] = ωy
        vtk["χx"] = χx
        vtk["χy"] = χy
    end
end

ε² = 1
Ux = 0
b(x, z) = x
bx(x, z) = 1
H(x) = 1 - x^2

# nz = 10
# h = 2/(2nz - 3)
# println("h = ", h)
# p = zeros(2*nz, 2)
# p[1, :] = [0 0]
# p[2, :] = [h 0]
# for i=2:nz-1
#     p[2i-1, :] = [0  -(2i-3)*h/2]
#     p[2i,   :] = [h  -(i-1)*h]
# end
# p[2nz-1, :] = [0 -(2nz-3)*h/2]
# p[2nz,   :] = [h -(2nz-3)*h/2]
# t = [i + j for i=1:2nz-2, j=0:2]
# e = [1, 2, 2nz-1, 2nz]
# col = FEGrid(p, t, e, 1)

g, cols, node_conns = gen_mesh(0.01)

b_cols = [b.(col.p[:, 1], col.p[:, 2]) for col ∈ cols]

# LHSs = [get_LHS(col_i=i) for i ∈ eachindex(cols)]
# RHSs = [get_RHS(col_i=i) for i ∈ eachindex(cols)]
# sols = [LHSs[i]\RHSs[i]  for i ∈ eachindex(cols)]

# plot_2D()

solve(col_i=40)