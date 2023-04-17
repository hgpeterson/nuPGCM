using nuPGCM
using WriteVTK
using HDF5
using Delaunay
using PyPlot
using SparseArrays
using LinearAlgebra
using ProgressMeter

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_3D_valign_mesh(geo, nref, H)
    # surface mesh
    g_sfc = FEGrid(1, "meshes/$geo/mesh$nref.h5")

    # will we need to tessellate?
    tessellate = !isfile("meshes/$geo/t_col_$(nref)_1.h5")
    # tessellate = true

    # x and y for convenience
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

    # z_cols
    z_cols = Vector{Vector{Float64}}(undef, g_sfc.np)

    # add points to p, e, and tri_to_p
    nzs = Int64[i ∈ g_sfc.e["bdy"] ? 1 : ceil(H(g_sfc.p[i, :])/h) for i=1:g_sfc.np]
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>Int64[], "bot"=>Int64[])
    np = 0
    for i=1:g_sfc.np
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0]
        else
            # z = range(-H(g_sfc.p[i, :]), 0, length=nz)
            z = -H(g_sfc.p[i, :])*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
        end

        # add to p
        p[np+1:np+nz, :] = [x[i]*ones(nz)  y[i]*ones(nz)  z]
        z_cols[i] = z

        # add to e
        e["bot"] = [e["bot"]; np + 1]
        e["sfc"] = [e["sfc"]; np + nz]

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
    sf = ShapeFunctions(order=1, dim=3)
    sfi = ShapeFunctionIntegrals(sf, sf)

    # columnwise and global tessellation
    g_cols = Vector{FEGrid}(undef, g_sfc.nt)
    t = Matrix{Int64}(undef, 0, 4) 
    @showprogress "Generating columns..." for k=1:g_sfc.nt
        # number of points in vertical for each vertex of sfc tri
        lens = length.(tri_to_p[k, :])

        # local p and e for col
        nodes_col = [tri_to_p[k, 1]; tri_to_p[k, 2]; tri_to_p[k, 3]]
        p_col = p[nodes_col, :]  
        e_sfc_col = [1, lens[1]+1, lens[1]+lens[2]+1]
        e_bot_col = [lens[1], lens[1]+lens[2], lens[1]+lens[2]+lens[3]]

        # either compute or load t for col
        if tessellate
            t_col = generate_t_col(geo, nref, k, p, tri_to_p, lens, nodes_col)
        else
            t_col = load_t_col(geo, nref, k)
        end

        # add to global t
        t = [t; nodes_col[t_col]]

        # create e_col dictionary
        e_col = Dict("sfc"=>e_sfc_col, "bot"=>e_bot_col)

        # save column data
        g_cols[k] = FEGrid(1, p_col, t_col, e_col, sf, sfi)

        # remove from bot if in sfc
        g_cols[k].e["bot"] = g_cols[k].e["bot"][findall(i -> g_cols[k].e["bot"][i] ∉ g_cols[k].e["sfc"], 1:size(g_cols[k].e["bot"], 1))]
    end

    g = FEGrid(1, p, t, e)

    return g_sfc, g, g_cols, z_cols, p_to_tri
end

function generate_t_col(geo, nref, k, p, tri_to_p, lens, nodes_col)
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

    save_t_col(geo, nref, k, t_col)

    return t_col
end

function save_t_col(geo, nref, k, t_col)
    h5open("meshes/$geo/t_col_$(nref)_$k.h5", "w") do file
        write(file, "t_col", t_col)
    end
end

function load_t_col(geo, nref, k)
    file = h5open("meshes/$geo/t_col_$(nref)_$k.h5", "r")
    t_col = read(file, "t_col")
    close(file)
    return t_col
end

"""
Solve
    -ε²∂zz(ωˣ) - ωʸ =  ∂y(b),
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
      -∂zz(χˣ) - ωˣ = 0,
      -∂zz(χʸ) - ωʸ = 0,
with bc
    z = 0:   ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    z = -H:  χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
"""
function solve_baroclinic_1dfe(z, bx, by, Ux, Uy, τx, τy, ε², f)
    # create 1D grid
    nz = size(z, 1)
    p = reshape(z, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nz])
    g = FEGrid(1, p, t, e)

    # indices
    ωxmap = 0*g.np+1:1*g.np
    ωymap = 1*g.np+1:2*g.np
    χxmap = 2*g.np+1:3*g.np
    χymap = 3*g.np+1:4*g.np
    N = 4*g.np

    # unpack
    J = g.J
    s = g.sfi

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g.nt
        # stiffness and mass matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # RHS
        if size(bx, 1) == g.nt
            # b is linear
            r[ωxmap[g.t[k, :]]] += by[k]*M*[1, 1]
            r[ωymap[g.t[k, :]]] -= bx[k]*M*[1, 1]
        elseif size(bx, 1) == 2g.nt
            # b is quadratic
            r[ωxmap[g.t[k, :]]] += M*[by[2k-1], by[2k]]
            r[ωymap[g.t[k, :]]] -= M*[bx[2k-1], bx[2k]]
        end

        # indices
        ωxi = ωxmap[g.t[k, :]]
        ωyi = ωymap[g.t[k, :]]
        χxi = χxmap[g.t[k, :]]
        χyi = χymap[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ≠ 1 &&  g.t[k, i] ≠ nz
                # -ε²∂zz(ωx)
                push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
                # -ωy
                push!(A, (ωxi[i], ωyi[j], -f*M[i, j]))

                # -ε²∂zz(ωy)
                push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
                # +ωx
                push!(A, (ωyi[i], ωxi[j], f*M[i, j]))
            end
            if g.t[k, i] ≠ nz
                # -∂zz(χx)
                push!(A, (χxi[i], χxi[j], K[i, j]))
                # -ωx
                push!(A, (χxi[i], ωxi[j], -M[i, j]))

                # -∂zz(χy)
                push!(A, (χyi[i], χyi[j], K[i, j]))
                # -ωy
                push!(A, (χyi[i], ωyi[j], -M[i, j]))
            end
        end
    end

    # z = -H: χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
    push!(A, (ωxmap[1], χxmap[1], 1))
    push!(A, (ωymap[1], χymap[1], 1))
    r[ωxmap[1]] = 0
    r[ωymap[1]] = 0

    # z = 0: ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    push!(A, (ωxmap[nz], ωxmap[nz], 1))
    push!(A, (ωymap[nz], ωymap[nz], 1))
    push!(A, (χxmap[nz], χxmap[nz], 1))
    push!(A, (χymap[nz], χymap[nz], 1))
    r[ωxmap[nz]] = -τy/ε²
    r[ωymap[nz]] = τx/ε²
    r[χxmap[nz]] = Uy
    r[χymap[nz]] = -Ux

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # solve
    sol = A\r
    return sol
end

function get_ω_U(g_sfc, g, z_cols, H, ε², f; showplots=false)
    # solve for ω_Uˣ
    ωx_Ux = zeros(g.np)
    ωy_Ux = zeros(g.np)
    χx_Ux = zeros(g.np)
    χy_Ux = zeros(g.np)
    j = 0
    for i ∈ eachindex(z_cols)
        nz = size(z_cols[i], 1)
        if nz == 1
            j += nz
            continue
        end
        x = g_sfc.p[i, :]
        sol = solve_baroclinic_1dfe(z_cols[i], zeros(nz-1), zeros(nz-1), H(x)^2, 0, 0, 0, ε², f(x))
        ωx_Ux[j+1:j+nz] = sol[0*nz+1:1*nz]
        ωy_Ux[j+1:j+nz] = sol[1*nz+1:2*nz]
        χx_Ux[j+1:j+nz] = sol[2*nz+1:3*nz]
        χy_Ux[j+1:j+nz] = sol[3*nz+1:4*nz]
        j += nz
    end

    if showplots
        ωx_Ux_bot = FEField(ωx_Ux[g.e["bot"]], g_sfc)
        ωy_Ux_bot = FEField(ωy_Ux[g.e["bot"]], g_sfc)
        quick_plot(ωx_Ux_bot, L"\omega^x_{U^x}(-H)", "scratch/images/omegax_Ux.png")
        quick_plot(ωy_Ux_bot, L"\omega^y_{U^x}(-H)}", "scratch/images/omegay_Ux.png")
        write_vtk(g, "output/baroclinic_Ux.vtu", Dict("ωx_Ux"=>ωx_Ux, "ωy_Ux"=>ωy_Ux, "χx_Ux"=>χx_Ux, "χy_Ux"=>χy_Ux))
    end

    return ωx_Ux, ωy_Ux, χx_Ux, χy_Ux
end

function get_ω_τ(g_sfc, g, z_cols, H, ε², f; showplots=false)
    # solve for ω_τˣ
    ωx_τx = zeros(g.np)
    ωy_τx = zeros(g.np)
    χx_τx = zeros(g.np)
    χy_τx = zeros(g.np)
    j = 0
    for i ∈ eachindex(z_cols)
        nz = size(z_cols[i], 1)
        if nz == 1
            j += nz
            continue
        end
        x = g_sfc.p[i, :]
        sol = solve_baroclinic_1dfe(z_cols[i], zeros(nz-1), zeros(nz-1), 0, 0, H(x)^2, 0, ε², f(x))
        ωx_τx[j+1:j+nz] = sol[0*nz+1:1*nz]
        ωy_τx[j+1:j+nz] = sol[1*nz+1:2*nz]
        χx_τx[j+1:j+nz] = sol[2*nz+1:3*nz]
        χy_τx[j+1:j+nz] = sol[3*nz+1:4*nz]
        j += nz
    end
    
    if showplots
        ωx_τx_bot = FEField(ωx_τx[g.e["bot"]], g_sfc)
        ωy_τx_bot = FEField(ωy_τx[g.e["bot"]], g_sfc)
        quick_plot(ωx_τx_bot, L"\omega^x_{\tau^x}(-H)", "scratch/images/omegax_taux.png")
        quick_plot(ωy_τx_bot, L"\omega^y_{\tau^x}(-H)}", "scratch/images/omegay_taux.png")
        write_vtk(g, "output/baroclinic_taux.vtu", Dict("ωx_τx"=>ωx_τx, "ωy_τx"=>ωy_τx, "χx_τx"=>χx_τx, "χy_τx"=>χy_τx))
    end

    return ωx_τx, ωy_τx, χx_τx, χy_τx
end

function get_ω_b(g_sfc, g, g_cols, z_cols, p_to_tri, ε², f, b; showplots=false)
    # grid
    nzs = [size(col, 1) for col ∈ z_cols]

    # b must be second order
    sf2 = ShapeFunctions(order=2, dim=3)
    sfi2 = ShapeFunctionIntegrals(sf2, sf2)
    b_cols = [FEGrid(2, col.p, col.t, col.e, sf2, sfi2) for col ∈ g_cols] # even this takes a while!

    # setup arrays
    bx = [zeros(2nz-2) for nz ∈ nzs]
    by = [zeros(2nz-2) for nz ∈ nzs]
    @showprogress "Computing buoyancy gradients..." for k=1:g_sfc.nt
        b_col = FEField(b, b_cols[k])
        n = 0
        for i=1:3
            ig = g_sfc.t[k, i]
            x = g_sfc.p[ig, 1]
            y = g_sfc.p[ig, 2]
            weight = 1/size(p_to_tri[ig], 1)
            for j=1:nzs[ig]-1
                # maybe store these k_tets! represent as a matrix?
                k_tet = findfirst(k_tet -> n+j ∈ g_cols[k].t[k_tet, :] && n+j+1 ∈ g_cols[k].t[k_tet, :], 1:g_cols[k].nt)
                bx[ig][2j-1] += weight*∂x(b_col, [x, y, z_cols[ig][j]], k_tet)
                bx[ig][2j]   += weight*∂x(b_col, [x, y, z_cols[ig][j+1]], k_tet)
                by[ig][2j-1] += weight*∂y(b_col, [x, y, z_cols[ig][j]], k_tet)
                by[ig][2j]   += weight*∂y(b_col, [x, y, z_cols[ig][j+1]], k_tet)
            end
            n += nzs[ig]
        end
    end
    # println("finding edges")
    # emap, edges, bndix = all_edges(g.t)
    # midpts = 1/2*reshape(g.p[edges[:, 1], :] + g.p[edges[:, 2], :], (size(edges, 1), :))
    # p2 = [g.p; midpts]
    # t2 = hcat(g.t, g.np .+ emap)
    # sf2 = ShapeFunctions(order=2, dim=3)
    # println("forming derivative matrices")
    # p1_ref = reference_element_nodes(1, 3)
    # Dξ = [∂φ(sf2, j, 1, p1_ref[i, :]) for i=1:4, j=1:10]
    # Dη = [∂φ(sf2, j, 2, p1_ref[i, :]) for i=1:4, j=1:10]
    # Dζ = [∂φ(sf2, j, 3, p1_ref[i, :]) for i=1:4, j=1:10]
    # b2 = [b(p2[t2[k, i], :]) for i=1:10, k=1:g.nt]
    # bξ = Dξ*b2
    # bη = Dη*b2
    # bζ = Dζ*b2
    # bx = bξ.*g.J.Js[:, 1, 1]' + bη.*g.J.Js[:, 2, 1]' + bζ.*g.J.Js[:, 3, 1]'
    # by = bξ.*g.J.Js[:, 1, 2]' + bη.*g.J.Js[:, 2, 2]' + bζ.*g.J.Js[:, 3, 2]'
    # nzs = [size(col, 1) for col ∈ z_cols]
    # col_indices = [sum(nzs[1:i-1])+1:sum(nzs[1:i-1])+nzs[i] for i ∈ eachindex(nzs)]
    # bx_cols = [zeros(2nz-2) for nz ∈ nzs]
    # by_cols = [zeros(2nz-2) for nz ∈ nzs]
    # @showprogress "Computing buoyancy gradients" for i_col ∈ eachindex(col_indices)
    #     weight = 1/size(p_to_tri[i_col], 1)
    #     for j=1:nzs[i_col]-1
    #         k_tet = findfirst(k -> col_indices[i_col][j] ∈ g.t[k, :] && col_indices[i_col][j+1] ∈ g.t[k, :], 1:g.nt)
    #         i1_tet = findfirst(i -> g.t[k_tet, i] == col_indices[i_col][j], 1:g.nn)
    #         i2_tet = findfirst(i -> g.t[k_tet, i] == col_indices[i_col][j+1], 1:g.nn)
    #         bx_cols[i_col][2j-1] += weight*bx[i1_tet, k_tet]
    #         bx_cols[i_col][2j]   += weight*bx[i2_tet, k_tet]
    #         by_cols[i_col][2j-1] += weight*by[i1_tet, k_tet]
    #         by_cols[i_col][2j]   += weight*by[i2_tet, k_tet]
    #     end
    # end

    # solve 
    ωx_b = zeros(g.np)
    ωy_b = zeros(g.np)
    χx_b = zeros(g.np)
    χy_b = zeros(g.np)
    j = 0
    for i ∈ eachindex(z_cols)
        nz = nzs[i]
        if nz ≤ 2
            j += nz
            continue
        end
        x = g_sfc.p[i, :]
        sol = solve_baroclinic_1dfe(z_cols[i], bx[i], by[i], 0, 0, 0, 0, ε², f(x))
        ωx_b[j+1:j+nz] = sol[0*nz+1:1*nz]
        ωy_b[j+1:j+nz] = sol[1*nz+1:2*nz]
        χx_b[j+1:j+nz] = sol[2*nz+1:3*nz]
        χy_b[j+1:j+nz] = sol[3*nz+1:4*nz]
        j += nz
    end

    if showplots
        ωx_b_bot = FEField(ωx_b[g.e["bot"]], g_sfc)
        ωy_b_bot = FEField(ωy_b[g.e["bot"]], g_sfc)
        quick_plot(ωx_b_bot, L"\omega^x_b(-H)", "scratch/images/omegax_b.png")
        quick_plot(ωy_b_bot, L"\omega^y_b(-H)}", "scratch/images/omegay_b.png")
        write_vtk(g, "output/baroclinic_b.vtu", Dict("ωx_b"=>ωx_b, "ωy_b"=>ωy_b, "χx_b"=>χx_b, "χy_b"=>χy_b))
    end

    return ωx_b, ωy_b, χx_b, χy_b
end

### 

function test_1d()
    ε² = 1e-4
    ε = sqrt(ε²)
    nz = 2^8
    H = 1
    z = @. -H*(cos(π*(0:nz-1)/(nz-1)) + 1)/2
    bx = zeros(nz-1)
    by = zeros(nz-1)
    Ux = 1
    Uy = 0
    τx = 0
    τy = 0
    f = 1

    # numerical sol
    sol = solve_baroclinic_1dfe(z, bx, by, Ux, Uy, τx, τy, ε², f)
    ωx = sol[1:nz]
    ωy = sol[nz+1:2nz]
    χx = sol[2nz+1:3nz]
    χy = sol[3nz+1:4nz]

    # BL sol

    q = sqrt(f)
    z_B = @. (z + H)/ε
    c = 1/(q*H)
    χx_I = 0
    χy_I = @. -(z + H)/H
    χx_B = zeros(nz)
    χy_B = @. c*exp(-q*z_B)*sin(q*z_B)
    χx_BL = χx_I .+ ε*χx_B
    χy_BL = χy_I .+ ε*χy_B
    ωx_I = 0
    ωy_I = 0
    ωx_B = zeros(nz)
    ωy_B = @. 2*c*q^2*exp(-q*z_B)*cos(q*z)
    ωx_BL = ωx_I .+ 1/ε*ωx_B
    ωy_BL = ωy_I .+ 1/ε*ωy_B

    # q = sqrt(f)
    # z_B = (z .+ H)/ε
    # c1 = -q^2*(H*q - ε)/(ε^3 - 2*H*q*ε^2 + 2*H^2*q^2*ε)
    # c2 = H*q^3/(ε^3 - 2*H*q*ε^2 + 2*H^2*q^2*ε)
    # ωx_I = 0
    # ωy_I = 0
    # ωx_B = @. exp(-q*z_B)*(c1*cos(q*z_B) + c2*sin(q*z_B))
    # ωy_B = @. 2*exp(-q*z_B)*(c2*cos(q*z_B) - c1*sin(q*z_B))
    # ωx_BL = ωx_I .+ ωx_B
    # ωy_BL = ωy_I .+ ωy_B
    # χx_I = -ε²*c2/(2*H*q^2)*z
    # χy_I = -1 .+ (ε²*c1/q^2 - 1)/H*z
    # χx_B = @. ε²/(2*q^2)*exp(-q*z_B)*(-c2*cos(q*z_B) + c1*sin(q*z_B))
    # χy_B = @. ε²/q^2*exp(-q*z_B)*(c1*cos(q*z_B) + c2*sin(q*z_B))
    # χx_BL = χx_I .+ χx_B
    # χy_BL = χy_I .+ χy_B

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(3.2, 5.2))
    ax[1, 1].plot(ωx, z, label=L"\omega^x")
    ax[1, 1].plot(ωy, z, label=L"\omega^y")
    ax[1, 1].plot(ωx_BL, z, "k--", lw=0.5)
    ax[1, 1].plot(ωy_BL, z, "k--", lw=0.5)
    ax[1, 2].plot(χx, z, label=L"\chi^x")
    ax[1, 2].plot(χy, z, label=L"\chi^y")
    ax[1, 2].plot(χx_BL, z, "k--", lw=0.5)
    ax[1, 2].plot(χy_BL, z, "k--", lw=0.5)
    ax[2, 1].plot(ωx, z, label=L"\omega^x")
    ax[2, 1].plot(ωy, z, label=L"\omega^y")
    ax[2, 1].plot(ωx_BL, z, "k--", lw=0.5)
    ax[2, 1].plot(ωy_BL, z, "k--", lw=0.5)
    ax[2, 2].plot(χx, z, label=L"\chi^x")
    ax[2, 2].plot(χy, z, label=L"\chi^y")
    ax[2, 2].plot(χx_BL, z, "k--", lw=0.5)
    ax[2, 2].plot(χy_BL, z, "k--", lw=0.5)
    ax[1, 1].set_ylabel(L"z")
    ax[2, 1].set_ylabel(L"z")
    ax[2, 1].set_xlabel(L"\omega")
    ax[2, 2].set_xlabel(L"\chi")
    ax[1, 1].legend()
    ax[1, 2].legend()
    # ax[2, 1].set_ylim(-1, -1 + 10*ε/sqrt(f))
    # ax[2, 1].set_ylim(-10*ε/sqrt(f), 0)
    ax[2, 2].set_xlim(-0.1, 0.01)
    # ax[2, 2].set_xlim(-1.01, -0.99)
    # ax[2, 2].set_ylim(-1, -1 + 10*ε/sqrt(f))
    ax[2, 2].set_ylim(-10*ε/sqrt(f), 0)
    ax[1, 2].set_yticklabels([])
    ax[2, 2].set_yticklabels([])
    savefig("scratch/images/omega_chi.png")
    println("scratch/images/omega_chi.png")
    plt.close()
end

# test_1d()

### 