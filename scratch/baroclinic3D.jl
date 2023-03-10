using nuPGCM
using HDF5
using PyPlot
using Printf
using ProgressMeter

include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_mesh(ifile, H; order)
    # load mesh of circle
    file = h5open(ifile, "r")
    p = read(file, "p")
    t = Int64.(read(file, "t"))
    e = Int64.(read(file, "e")[:, 1])
    close(file)
    e = Dict("coastline"=>e)
    g_sfc = FEGrid(1, p, t, e)
    x = g_sfc.p[:, 1]
    y = g_sfc.p[:, 2]

    # mesh res
    emap, edges, bndix = all_edges(g_sfc.t)
    h = 1/size(edges, 1)*sum(norm(g_sfc.p[edges[i, 1], :] - g_sfc.p[edges[i, 2], :]) for i in axes(edges, 1))

    # mapping from points to triangles:
    #   `p_to_tri[i]` is vector of cartesian indices pointing to where point `i` is in `g_sfc.t`
    p_to_tri = [findall(I -> i ∈ g_sfc.t[I], CartesianIndices(size(g_sfc.t))) for i=1:g_sfc.np]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    tri_to_p = [Int64[] for k=1:g_sfc.nt, i=1:3] # allocate

    # node_cols
    node_cols = Vector{Vector{Float64}}(undef, g_sfc.np)

    # add points to p, e, and tri_to_p
    nzs = Int64[i ∈ g_sfc.e["coastline"] ? 1 : ceil(H(x[i], y[i])/h) for i=1:g_sfc.np]
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>Int64[], "bot"=>Int64[])
    np = 0
    for i=1:g_sfc.np
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0]
        else
            z = -range(0, H(x[i], y[i]), length=nz)
        end

        # add to p
        p[np+1:np+nz, :] = [x[i]*ones(nz)  y[i]*ones(nz)  z]
        node_cols[i] = z

        # add to e
        e["sfc"] = [e["sfc"]; np + 1]
        if nz != 1
            e["bot"] = [e["bot"]; np + nz]
        end

        # add to tri_to_p
        for I ∈ p_to_tri[i]
            for j=np+1:np+nz
                push!(tri_to_p[I], j)
            end
        end

        # iterate
        np += nz
    end

    # setup shape functions and their integrals now since they're the same for each grid
    sf = ShapeFunctions(order=order, dim=3)
    sfi = ShapeFunctionIntegrals(sf, sf)

    # columnwise and global tessellation
    el_cols = Vector{FEGrid}(undef, g_sfc.nt)
    t = Matrix{Int64}(undef, 0, 4) 
    for k=1:g_sfc.nt
        # number of points in vertical for each vertex of sfc tri
        lens = length.(tri_to_p[k, :])

        # local p and e for column
        nodes_col = [tri_to_p[k, 1]; tri_to_p[k, 2]; tri_to_p[k, 3]]
        p_col = p[nodes_col, :]  
        e_sfc_col = [1, lens[1]+1, lens[1]+lens[2]+1]
        e_bot_col = [lens[1], lens[1]+lens[2], lens[1]+lens[2]+lens[3]]

        # start local t
        t_col = Matrix{Int64}(undef, 0, 4) 

        # first top tri is at sfc
        top = [tri_to_p[k, i][1] for i=1:3]

        # continue down to bottom
        for j=2:maximum(lens)
            # make bottom tri from next nodes down or top tri nodes
            bot = [j ≤ lens[i] ? tri_to_p[k, i][j] : top[i] for i=1:3]

            # use delaunay to tessellate
            ig = unique(vcat(top, bot))
            tl = delaunay(p[ig, :]).simplices

            # add to t
            t = [t; ig[tl]]

            # add to t_col
            i_col = Int64.(indexin(ig, nodes_col))
            t_col = [t_col; i_col[tl]]

            # continue
            top = bot
        end

        # create e_col dictionary
        e_col = Dict("sfc"=>e_sfc_col, "bot"=>e_bot_col)

        # save column data
        el_cols[k] = FEGrid(order, p_col, t_col, e_col, sf, sfi)

        # remove from bot if in sfc
        el_cols[k].e["bot"] = el_cols[k].e["bot"][findall(i -> el_cols[k].e["bot"][i] ∉ el_cols[k].e["sfc"], 1:size(el_cols[k].e["bot"], 1))]
    end

    g = FEGrid(order, p, t, e)

    return g_sfc, g, el_cols, node_cols, p_to_tri
end

function main()
    # params
    ε² = 1

    # functions
    H(x, y) = 1 - x^2 - y^2
    Hx(x, y) = -2*x
    Hy(x, y) = -2*y
    τx(x, y) = 0
    τy(x, y) = 0
    Ux(x, y) = 0
    Uy(x, y) = 0
    # Ux(x, y) = H(x, y)^2
    # Uy(x, y) = H(x, y)^2
    # b(x, y, z) = 0
    # bx(x, y, z) = 0
    # by(x, y, z) = 0
    # b(x, y, z) = x
    # bx(x, y, z) = 1
    # by(x, y, z) = 0
    # b(x, y, z) = x^3 + y^3
    # bx(x, y, z) = 3x^2
    # by(x, y, z) = 3y^2
    δ = 0.1
    b(x, y, z) = z + δ*exp(-(z + H(x, y))/δ)
    bx(x, y, z) = -Hx(x, y)*exp(-(z + H(x, y))/δ)
    by(x, y, z) = -Hy(x, y)*exp(-(z + H(x, y))/δ)

    # # analytical solution
    # ωx_a(x, y, z) = z*exp(z)*cos(x)*sin(y)
    # ωy_a(x, y, z) = z*exp(z)*cos(y)*sin(x)
    # Ux(x, y) = -(2 - exp(-H(x, y))*(2 + 2*H(x, y) + H(x, y)^2))*cos(y)*sin(x)
    # Uy(x, y) =  (2 - exp(-H(x, y))*(2 + 2*H(x, y) + H(x, y)^2))*cos(x)*sin(y)
    # τx(x, y) = ωy_a(x, y, 0)
    # τy(x, y) = -ωx_a(x, y, 0)
    # b(x, y, z) = -exp(z)*(z*sin(x)*sin(y) - ε²*(z + 2)*cos(x)*cos(y))

    # grid
    g_sfc, g, el_cols, node_cols, p_to_tri = gen_mesh("meshes/circle/mesh3.h5", H, order=1)
    println("nel_cols = ", size(el_cols, 1))
    nzs = [size(col, 1) for col ∈ node_cols]
    # el_cols2 = [FEGrid(2, col) for col ∈ el_cols]
    el_cols2 = []
    @showprogress for col ∈ el_cols
        push!(el_cols2, FEGrid(2, col))
    end
    # el_cols2 = [FEGrid(2, col) for col ∈ el_cols]

    # evaluate functions for each column
    # b_el_cols = [b.(col.p[:, 1], col.p[:, 2], col.p[:, 3]) for col ∈ el_cols]
    b_el_cols = [b.(col.p[:, 1], col.p[:, 2], col.p[:, 3]) for col ∈ el_cols2]
    Ux_node_cols = Ux.(g_sfc.p[:, 1], g_sfc.p[:, 2])
    Uy_node_cols = Uy.(g_sfc.p[:, 1], g_sfc.p[:, 2])
    τx_node_cols = τx.(g_sfc.p[:, 1], g_sfc.p[:, 2])
    τy_node_cols = τy.(g_sfc.p[:, 1], g_sfc.p[:, 2])
    # bx_node_cols = [zeros(nz-1) for nz ∈ nzs]
    # by_node_cols = [zeros(nz-1) for nz ∈ nzs]
    bx_node_cols = [zeros(2nz-2) for nz ∈ nzs]
    by_node_cols = [zeros(2nz-2) for nz ∈ nzs]
    for k=1:g_sfc.nt
        # b_col = FEField(b_el_cols[k], el_cols[k])
        b_col = FEField(b_el_cols[k], el_cols2[k])
        n = 0
        for i=1:3
            ig = g_sfc.t[k, i]
            x = g_sfc.p[ig, 1]
            y = g_sfc.p[ig, 2]
            weight = 1/size(p_to_tri[ig], 1)
            for j=1:nzs[ig]-1
                k_tet = findfirst(k_tet -> n+j ∈ el_cols[k].t[k_tet, :] && n+j+1 ∈ el_cols[k].t[k_tet, :], 1:el_cols[k].nt)
                # bx_node_cols[ig][j] += weight*∂x(b_col, [0, 0, 0], k_tet)
                # by_node_cols[ig][j] += weight*∂y(b_col, [0, 0, 0], k_tet)
                bx_node_cols[ig][2j-1] += weight*∂x(b_col, [x, y, node_cols[ig][j]], k_tet)
                bx_node_cols[ig][2j]   += weight*∂x(b_col, [x, y, node_cols[ig][j+1]], k_tet)
                by_node_cols[ig][2j-1] += weight*∂y(b_col, [x, y, node_cols[ig][j]], k_tet)
                by_node_cols[ig][2j]   += weight*∂y(b_col, [x, y, node_cols[ig][j+1]], k_tet)
            end
            n += nzs[ig]
        end
    end

    # solve
    sols = [nzs[i] == 1 ? [0.0, 0.0] : solve_baroclinic_1dfe(node_cols[i], bx_node_cols[i], by_node_cols[i], Ux_node_cols[i], Uy_node_cols[i], τx_node_cols[i], τy_node_cols[i], ε²) for i ∈ eachindex(node_cols)]
    plot_3D(g, sols)
end

main()

println("Done.")