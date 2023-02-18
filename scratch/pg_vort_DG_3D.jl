## Solve
##     -ε²∂zz(ωˣ) - ωʸ =  ∂y(b),
##     -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
##        ∂zz(χˣ) + ωˣ = 0,
##        ∂zz(χʸ) + ωʸ = 0,
## with bc
## At z = 0:
##     • ωˣ = 0, ωʸ = 0, χˣ = Uʸ, χʸ = -Uˣ
## At z = -H:
##     • χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0

using nuPGCM
using WriteVTK
using HDF5
using Delaunay
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_mesh(ifile)
    # load mesh of circle
    file = h5open(ifile, "r")
    p_sfc = read(file, "p")
    t_sfc = Int64.(read(file, "t"))
    e_sfc = Int64.(read(file, "e")[:, 1])
    close(file)
    x = p_sfc[:, 1]
    y = p_sfc[:, 2]
    np_sfc = size(p_sfc, 1)
    nt_sfc = size(t_sfc, 1)
    ne_sfc = size(e_sfc, 1)

    # store an array of columns
    cols = Vector{FEGrid}(undef, nt_sfc)

    # mesh res
    emap, edges, bndix = all_edges(t_sfc)
    h = 1/size(edges, 1)*sum(norm(p_sfc[edges[i, 1], :] - p_sfc[edges[i, 2], :]) for i in axes(edges, 1))

    # loop over columns
    for k=1:nt_sfc
        # start with nodes at the surface
        p_col = hcat(x[t_sfc[k, :]], y[t_sfc[k, :]], zeros(3))
        e_col = [1, 2, 3]
        
        # add nodes in vertical if not on coastline
        for i=1:3
            if t_sfc[k, i] ∉ e_sfc
                depth = H(x[t_sfc[k, i]], y[t_sfc[k, i]])
                nz = Int64(ceil(depth/h))
                z = -range(0, depth, length=nz)[2:end] # remove sfc node

                # add to p
                p_col = vcat(p_col, [x[t_sfc[k, i]]*ones(nz-1)  y[t_sfc[k, i]]*ones(nz-1)  z])

                # add to e
                e_col = [e_col; size(p_col, 1)]
            end
        end

        # mesh
        t_col = delaunay(p_col).simplices

        # save column
        cols[k] = FEGrid(p_col, t_col, e_col, 1)
    end

    p = cols[1].p
    t = cols[1].t
    e = cols[1].e
    for k=2:size(cols, 1)
        np = size(p, 1)
        p = [p; cols[k].p]
        t = [t; np .+ cols[k].t]
        e = [e; np .+ cols[k].e]
    end

    return cols, p, t, e
end

function var_indices(col)
    ωxmap = 0*col.np+1:1*col.np
    ωymap = 1*col.np+1:2*col.np
    χxmap = 2*col.np+1:3*col.np
    χymap = 3*col.np+1:4*col.np
    return ωxmap, ωymap, χxmap, χymap
end

function sfc_and_bot(col)
    perm = sortperm(col.p[col.e, 2], rev=true)
    if mod(size(perm, 1), 2) == 1
        # corner node → put it on the surface
        half = Int64(ceil(size(perm, 1)/2)) 
    else
        half = Int64(size(perm, 1)/2)
    end
    return col.e[perm[1:half]], col.e[perm[half+1:end]]
end

function get_LHS(col)
    # indices
    ωxmap, ωymap, χxmap, χymap = var_indices(col)
    N = 4*col.np

    # surface and bottom nodes
    sfc, bot = sfc_and_bot(col)

    # for element matricies
    col1 = FEGrid(col, 1)
    J = Jacobians(col1)
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
        push!(A, (ωxmap[i], χxmap[i], 1))
        push!(A, (ωymap[i], χymap[i], 1))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    return lu(A)
end

function get_RHS(col, col_i)
    # indices
    ωxmap, ωymap, χxmap, χymap = var_indices(col)
    N = 4*col.np

    # surface and bottom nodes
    sfc, bot = sfc_and_bot(col)

    # for surface integrals
    s = ShapeFunctionIntegrals(col.s, col.s)
    J = Jacobians(col)

    # for edge integrals
    w, ξ = quad_weights_points(deg=2*col.order, dim=1)
    s1D = ShapeFunctions(order=col.order, dim=1)

    # # stamp
    # r = zeros(N)
    # for k=1:col.nt
    #     # triangle nodes
    #     p_tri = col.p[col.t[k, 1:3], :]

    #     # which edge is vertical edge
    #     ie, edge = vert_edge(p_tri)
    #     node1 = col.t[k, edge[1]]
    #     node2 = col.t[k, edge[2]]

    #     # is it on left or right? +1 if right, -1 if left
    #     sign_multiplier = side_of_vert_edge(p_tri, ie)

    #     # z-coords of edge nodes
    #     z1 = col.p[node1, 2]
    #     z2 = col.p[node2, 2]

    #     # connected nodes in neighboring column 
    #     col_j, conn_node1 = node_conns[col_i][node1]
    #     col_j, conn_node2 = node_conns[col_i][node2]

    #     # average b values        
    #     b1 = (b_cols[col_i][node1] + b_cols[col_j][conn_node1])/2
    #     b2 = (b_cols[col_i][node2] + b_cols[col_j][conn_node2])/2

    #     # -∫_∂Ωᵢₑ bφᵢ dz 
    #     for i=1:2
    #         f(t) = (b1*φ(s1D, 1, t) + b2*φ(s1D, 2, t))*φ(s1D, i, t)
    #         ∫f = dot(w, f.(ξ))*abs(z2 - z1)/2
    #         r[ωymap[col.t[k, edge[i]]]] -= sign_multiplier*∫f
    #     end

    #     # +∫_Ω b∂x(φ) dxdz
    #     Cx = J.dets[k]*sum(s.CT.*J.Js[k, :, 1], dims=1)[1, :, :]
    #     r[ωymap[col.t[k, :]]] += Cx*b_cols[col_i][col.t[k, :]]
    # end

    # surface nodes 
    for i ∈ sfc
        r[ωxmap[i]] = 0
        r[ωymap[i]] = 0
        r[χxmap[i]] = Uy
        r[χymap[i]] = -Ux
    end

    # bottom nodes
    for i ∈ bot
        r[ωxmap[i]] = 0
        r[ωymap[i]] = 0
    end

    return r
end

function fd_sol(z, bx, by, ε², Ux, Uy)
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

        # eqtn 1: -ε²∂zz(ωˣ) - ωʸ = ∂y(b)
        # term 1
        push!(A, (ωxmap[j], ωxmap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωxmap[j], ωxmap[j],   -ε²*fd_zz[2]))
        push!(A, (ωxmap[j], ωxmap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωxmap[j], ωymap[j], -1))
        # rhs
        r[ωxmap[j]] = by[j]

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

    # ∫ zωˣ dz = Uy
    for j=1:nz-1
        push!(A, (ωxmap[1], ωxmap[j],     z[j]*(z[j+1] - z[j])/2))
        push!(A, (ωxmap[1], ωxmap[j+1], z[j+1]*(z[j+1] - z[j])/2))
    end
    r[ωxmap[1]] = Uy

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

ε² = 0.01
Ux(x, y) = 0
Uy(x, y) = 0
b(x, y, z) = x + y
bx(x, y, z) = 1
by(x, y, z) = 1
H(x, y) = 1 - x^2 - y^2

cols, p, t, e = gen_mesh("meshes/circle/mesh2.h5")
cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i ∈ axes(t, 1)]
vtk_grid("output/pg_vort_DG_3D.vtu", p', cells) do vtk
end
println("output/pg_vort_DG_3D.vtu")