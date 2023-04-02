using nuPGCM
using HDF5
using PyPlot
using Printf
using ProgressMeter

include("baroclinic.jl")
include("utils.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_3D_valign_mesh(g_sfc, H; order)
    # save x and y for convenience
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
    nzs = Int64[i ∈ g_sfc.e["bdy"] ? 1 : ceil(H(x[i], y[i])/h) for i=1:g_sfc.np]
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
        # if nz != 1
        #     e["bot"] = [e["bot"]; np + nz]
        # end
        if nz == 1
            e["bot"] = [e["bot"]; np + 1]
        else
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
    @showprogress "Generating columns..." for k=1:g_sfc.nt
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

            # add to t_col
            i_col = Int64.(indexin(ig, nodes_col))
            t_col = [t_col; i_col[tl]]

            # continue
            top = bot
        end

        # add to t
        t = [t; nodes_col[t_col]]

        # create e_col dictionary
        e_col = Dict("sfc"=>e_sfc_col, "bot"=>e_bot_col)

        # save column data
        el_cols[k] = FEGrid(order, p_col, t_col, e_col, sf, sfi)

        # remove from bot if in sfc
        el_cols[k].e["bot"] = el_cols[k].e["bot"][findall(i -> el_cols[k].e["bot"][i] ∉ el_cols[k].e["sfc"], 1:size(el_cols[k].e["bot"], 1))]
    end

    g = FEGrid(order, p, t, e)

    return g, el_cols, node_cols, p_to_tri
end

function get_ω_U(g_sfc, g, node_cols, H, ε², f)
    # solve for ω_Uˣ
    ωx_Ux = zeros(g.np)
    ωy_Ux = zeros(g.np)
    χx_Ux = zeros(g.np)
    χy_Ux = zeros(g.np)
    j = 0
    @showprogress "Solving..." for i ∈ eachindex(node_cols)
        nz = size(node_cols[i], 1)
        if nz == 1
            j += nz
            continue
        end
        x = g_sfc.p[i, 1]
        y = g_sfc.p[i, 2]
        sol = solve_baroclinic_1dfe(node_cols[i], zeros(nz-1), zeros(nz-1), H(x, y)^2, 0, 0, 0, ε², f(y))
        ωx_Ux[j+1:j+nz] = sol[0*nz+1:1*nz]
        ωy_Ux[j+1:j+nz] = sol[1*nz+1:2*nz]
        χx_Ux[j+1:j+nz] = sol[2*nz+1:3*nz]
        χy_Ux[j+1:j+nz] = sol[3*nz+1:4*nz]
        j += nz
    end

    # symmetry
    ωx_Uy = -ωy_Ux
    ωy_Uy = ωx_Ux
    χx_Uy = -χy_Ux
    χy_Uy = χx_Ux

    # plot
    ωx_Ux_bot = FEField(ωx_Ux[g.e["bot"]], g_sfc)
    ωy_Ux_bot = FEField(ωy_Ux[g.e["bot"]], g_sfc)
    quick_plot(ωx_Ux_bot, L"\omega^x_{U^x}(-H)", "scratch/images/omegax_Ux.png")
    quick_plot(ωy_Ux_bot, L"\omega^y_{U^x}(-H)}", "scratch/images/omegay_Ux.png")
    ωx_Uy_bot = FEField(ωx_Uy[g.e["bot"]], g_sfc)
    ωy_Uy_bot = FEField(ωy_Uy[g.e["bot"]], g_sfc)

    return ωx_Ux_bot, ωy_Ux_bot, ωx_Uy_bot, ωy_Uy_bot
end

function get_ω_τ(g_sfc, g, node_cols, ε², f)
    # solve for ω_τˣ
    ωx_τx = zeros(g.np)
    ωy_τx = zeros(g.np)
    χx_τx = zeros(g.np)
    χy_τx = zeros(g.np)
    j = 0
    @showprogress "Solving..." for i ∈ eachindex(node_cols)
        nz = size(node_cols[i], 1)
        if nz == 1
            j += nz
            continue
        end
        y = g_sfc.p[i, 2]
        sol = solve_baroclinic_1dfe(node_cols[i], zeros(nz-1), zeros(nz-1), 0, 0, 1, 0, ε², f(y))
        ωx_τx[j+1:j+nz] = sol[0*nz+1:1*nz]
        ωy_τx[j+1:j+nz] = sol[1*nz+1:2*nz]
        χx_τx[j+1:j+nz] = sol[2*nz+1:3*nz]
        χy_τx[j+1:j+nz] = sol[3*nz+1:4*nz]
        j += nz
    end

    # symmetry
    ωx_τy = -ωy_τx
    ωy_τy =  ωx_τx
    χx_τy = -χy_τx
    χy_τy =  χx_τx
    
    # plot
    ωx_τx_bot = FEField(ωx_τx[g.e["bot"]], g_sfc)
    ωy_τx_bot = FEField(ωy_τx[g.e["bot"]], g_sfc)
    quick_plot(ωx_τx_bot, L"\omega^x_{\tau^x}(-H)", "scratch/images/omegax_taux.png")
    quick_plot(ωy_τx_bot, L"\omega^y_{\tau^x}(-H)}", "scratch/images/omegay_taux.png")
    ωx_τy_bot = FEField(ωx_τy[g.e["bot"]], g_sfc)
    ωy_τy_bot = FEField(ωy_τy[g.e["bot"]], g_sfc)

    return ωx_τx_bot, ωy_τx_bot, ωx_τy_bot, ωy_τy_bot
end

# function get_ω_b(b, H, ε², g_sfc; b_order)
#     # grid
#     g, el_cols, node_cols, p_to_tri = gen_3D_valign_mesh(g_sfc, H, order=1)
#     println("nel_cols = ", size(el_cols, 1))
#     nzs = [size(col, 1) for col ∈ node_cols]
#     if b_order == 1
#         b_cols = el_cols
#     elseif b_order == 2
#         sf2 = ShapeFunctions(order=2, dim=3)
#         sfi2 = ShapeFunctionIntegrals(sf2, sf2)
#         b_cols = [FEGrid(2, col.p, col.t, col.e, sf2, sfi2) for col ∈ el_cols]
#     end

#     # evaluate functions for each column
#     b0 = [b.(col.p[:, 1], col.p[:, 2], col.p[:, 3]) for col ∈ b_cols]
#     if b_order == 1
#         bx0 = [zeros(nz-1) for nz ∈ nzs]
#         by0 = [zeros(nz-1) for nz ∈ nzs]
#     elseif b_order == 2
#         bx0 = [zeros(2nz-2) for nz ∈ nzs]
#         by0 = [zeros(2nz-2) for nz ∈ nzs]
#     end
#     @showprogress "Computing buoyancy gradients..." for k=1:g_sfc.nt
#         b_col = FEField(b0[k], b_cols[k])
#         n = 0
#         for i=1:3
#             ig = g_sfc.t[k, i]
#             x = g_sfc.p[ig, 1]
#             y = g_sfc.p[ig, 2]
#             weight = 1/size(p_to_tri[ig], 1)
#             # # compute weight based on angle
#             # v1 = g_sfc.p[g_sfc.t[k, mod1(i+1, 3)], :] - g_sfc.p[g_sfc.t[k, i], :]
#             # v2 = g_sfc.p[g_sfc.t[k, mod1(i+2, 3)], :] - g_sfc.p[g_sfc.t[k, i], :]
#             # weight = acos(dot(v1, v2)/norm(v1)/norm(v2))/2π
#             for j=1:nzs[ig]-1
#                 k_tet = findfirst(k_tet -> n+j ∈ el_cols[k].t[k_tet, :] && n+j+1 ∈ el_cols[k].t[k_tet, :], 1:el_cols[k].nt)
#                 if b_order == 1
#                     bx0[ig][j] += weight*∂x(b_col, [0, 0, 0], k_tet)
#                     by0[ig][j] += weight*∂y(b_col, [0, 0, 0], k_tet)
#                 elseif b_order == 2
#                     bx0[ig][2j-1] += weight*∂x(b_col, [x, y, node_cols[ig][j]], k_tet)
#                     bx0[ig][2j]   += weight*∂x(b_col, [x, y, node_cols[ig][j+1]], k_tet)
#                     by0[ig][2j-1] += weight*∂y(b_col, [x, y, node_cols[ig][j]], k_tet)
#                     by0[ig][2j]   += weight*∂y(b_col, [x, y, node_cols[ig][j+1]], k_tet)
#                 end
#             end
#             n += nzs[ig]
#         end
#     end

#     # solve 
#     ωx = zeros(g.np)
#     ωy = zeros(g.np)
#     χx = zeros(g.np)
#     χy = zeros(g.np)
#     j = 0
#     @showprogress "Solving..." for i ∈ eachindex(node_cols)
#         nz = nzs[i]
#         if nz ≤ 2
#             j += nz
#             continue
#         end
#         sol = solve_baroclinic_1dfe(node_cols[i], bx0[i], by0[i], 0, 0, 0, 0, ε²)
#         ωx[j+1:j+nz] = sol[0*nz+1:1*nz]
#         ωy[j+1:j+nz] = sol[1*nz+1:2*nz]
#         χx[j+1:j+nz] = sol[2*nz+1:3*nz]
#         χy[j+1:j+nz] = sol[3*nz+1:4*nz]
#         j += nz
#     end

#     # plot
#     b0 = b.(g.p[:, 1], g.p[:, 2], g.p[:, 3])
#     write_vtk(g, "output/baroclinic.vtu", Dict("ωx"=>ωx, "ωy"=>ωy, "χx"=>χx, "χy"=>χy, "b"=>b0))
# end