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

function gen_3D_valign_mesh(geo, nref, H; chebyshev=false, tessellate=nothing)
    # surface mesh
    g_sfc = FEGrid(1, "meshes/$geo/mesh$nref.h5")

    # will we need to tessellate?
    if tessellate === nothing
        tessellate = !isfile("meshes/$geo/t_col_$(nref)_1.h5")
        # tessellate = true
    end

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
            if chebyshev
                z = -H(g_sfc.p[i, :])*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
            else
                z = range(-H(g_sfc.p[i, :]), 0, length=nz)
            end
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
            # bx, by are constant discontinuous
            r[ωxmap[g.t[k, :]]] += by[k]*M*[1, 1]
            r[ωymap[g.t[k, :]]] -= bx[k]*M*[1, 1]
        elseif size(bx, 1) == 2g.nt
            # bx, by are linear discontinuous
            r[ωxmap[g.t[k, :]]] += M*[by[2k-1], by[2k]]
            r[ωymap[g.t[k, :]]] -= M*[bx[2k-1], bx[2k]]
        elseif size(bx, 1) == g.np
            # bx, by are linear continuous
            r[ωxmap[g.t[k, :]]] += M*by[g.t[k, :]]
            r[ωymap[g.t[k, :]]] -= M*bx[g.t[k, :]]
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

function get_ω_b(g_sfc, g, b_cols, z_cols, Dxs, Dys, ε², f, b; showplots=false)
    # grid
    nzs = [size(col, 1) for col ∈ z_cols]

    # setup arrays
    # bvals = [[b(b_cols[k].p[i, :]) for i=1:b_cols[k].np] for k=1:g_sfc.nt]
    # bx = [[Dxs[k][i]*bvals[k] for i=1:3] for k=1:g_sfc.nt]
    # by = [[Dys[k][i]*bvals[k] for i=1:3] for k=1:g_sfc.nt]
    bx = [zeros(2nz-2) for nz ∈ nzs]
    by = [zeros(2nz-2) for nz ∈ nzs]
    for k=1:g_sfc.nt
        b_col = FEField(b, b_cols[k])
        for i=1:3
            ig = g_sfc.t[k, i]
            bx[ig] += Dxs[k][i]*b_col.values
            by[ig] += Dys[k][i]*b_col.values
        end
    end

    # solve 
    # ωx_b = [[zeros(nzs[g_sfc.t[k, i]]) for i=1:3] for k=1:g_sfc.nt]
    # ωy_b = [[zeros(nzs[g_sfc.t[k, i]]) for i=1:3] for k=1:g_sfc.nt]
    # χx_b = [[zeros(nzs[g_sfc.t[k, i]]) for i=1:3] for k=1:g_sfc.nt]
    # χy_b = [[zeros(nzs[g_sfc.t[k, i]]) for i=1:3] for k=1:g_sfc.nt]
    # for k=1:g_sfc.nt
    #     for i=1:3
    #         ig = g_sfc.t[k, i]
    #         nz = nzs[ig]
    #         if nz ≤ 2
    #             continue
    #         end
    #         x = g_sfc.p[ig, :]
    #         sol = solve_baroclinic_1dfe(z_cols[ig], bx[k][i], by[k][i], 0, 0, 0, 0, ε², f(x))
    #         ωx_b[k][i] = sol[0*nz+1:1*nz]
    #         ωy_b[k][i] = sol[1*nz+1:2*nz]
    #         χx_b[k][i] = sol[2*nz+1:3*nz]
    #         χy_b[k][i] = sol[3*nz+1:4*nz]
    #     end
    # end
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

function get_b_gradient_matrices(b_col, g_col, g_sfc, z_cols, k) 
    p1_ref = reference_element_nodes(1, 3)
    Dξ = [∂φ(b_col.sf, j, 1, p1_ref[i, :]) for i=1:g_col.nn, j=1:b_col.nn]
    Dη = [∂φ(b_col.sf, j, 2, p1_ref[i, :]) for i=1:g_col.nn, j=1:b_col.nn]
    Dζ = [∂φ(b_col.sf, j, 3, p1_ref[i, :]) for i=1:g_col.nn, j=1:b_col.nn]
    Dxs = Vector{SparseMatrixCSC}(undef, 3)
    Dys = Vector{SparseMatrixCSC}(undef, 3)
    n = 0
    for i=1:3
        ig = g_sfc.t[k, i]
        nz = size(z_cols[ig], 1)
        weight = 1/size(p_to_tri[ig], 1)
        Dx = Tuple{Int64,Int64,Float64}[]
        Dy = Tuple{Int64,Int64,Float64}[]
        for j=1:nz-1
            k_tet = findfirst(k -> n+j ∈ g_col.t[k, :] && n+j+1 ∈ g_col.t[k, :], 1:g.nt)
            ξx = g_col.J.Js[k_tet, 1, 1]
            ξy = g_col.J.Js[k_tet, 1, 2]
            ηx = g_col.J.Js[k_tet, 2, 1]
            ηy = g_col.J.Js[k_tet, 2, 2]
            ζx = g_col.J.Js[k_tet, 3, 1]
            ζy = g_col.J.Js[k_tet, 3, 2]
            i1_tet = findfirst(i -> g_col.t[k_tet, i] == n+j, 1:g_col.nn) 
            i2_tet = findfirst(i -> g_col.t[k_tet, i] == n+j+1, 1:g_col.nn)
            for l=1:b_col.nn
                push!(Dx, (2j-1, b_col.t[k_tet, l], weight*(Dξ[i1_tet, l]*ξx + Dη[i1_tet, l]*ηx + Dζ[i1_tet, l]*ζx)))
                push!(Dx, (2j,   b_col.t[k_tet, l], weight*(Dξ[i2_tet, l]*ξx + Dη[i2_tet, l]*ηx + Dζ[i2_tet, l]*ζx)))
                push!(Dy, (2j-1, b_col.t[k_tet, l], weight*(Dξ[i1_tet, l]*ξy + Dη[i1_tet, l]*ηy + Dζ[i1_tet, l]*ζy)))
                push!(Dy, (2j,   b_col.t[k_tet, l], weight*(Dξ[i2_tet, l]*ξy + Dη[i2_tet, l]*ηy + Dζ[i2_tet, l]*ζy)))
            end
        end
        Dxs[i] = sparse((x -> x[1]).(Dx), (x -> x[2]).(Dx), (x -> x[3]).(Dx), 2nz-2, b_col.np)
        Dys[i] = sparse((x -> x[1]).(Dy), (x -> x[2]).(Dy), (x -> x[3]).(Dy), 2nz-2, b_col.np)
        n += nz
    end

    return Dxs, Dys
end

### 

function test_1d()
    # inputs
    ε² = 1e-4
    ε = sqrt(ε²)
    nz = 2^8
    H = 1
    z = @. -H*(cos(π*(0:nz-1)/(nz-1)) + 1)/2
    bx = z
    by = ones(nz)
    Ux = 0
    Uy = 0
    τx = 0
    τy = 0
    y = 1

    # numerical sol
    sol = solve_baroclinic_1dfe(z, bx, by, Ux, Uy, τx, τy, ε², y)
    ωx = sol[1:nz]
    ωy = sol[nz+1:2nz]
    χx = sol[2nz+1:3nz]
    χy = sol[3nz+1:4nz]

    # BL sol
    q = sqrt(y/2)
    z_b = (z .+ H)/ε
    z_s = z/ε

    # # transport
    # c1 = -q/H
    # c2 = +q/H
    # χx_I0 = 0
    # χy_I0 = @. -(z + H)/H
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))
    # χy_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωx_BL = 1/ε*ωx_B1
    # ωy_BL = 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ ε*(χx_I1 .+ χx_B1)
    # χy_BL = χy_I0 .+ ε*(χy_I1 .+ χy_B1)

    # # wind
    # c1 = c2 = -1/(2*H*q)
    # χx_I0 = @. (z + H)/(2*H*q^2)
    # χy_I0 = 0
    # ωx0_B0 = @. -exp(q*z_s)*sin(q*z_s)
    # ωy0_B0 = @. exp(q*z_s)*cos(q*z_s)
    # χx0_B0 = @. -1/(2*q^2)*exp(q*z_s)*cos(q*z_s)
    # χy0_B0 = @. -1/(2*q^2)*exp(q*z_s)*sin(q*z_s)
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))
    # χy_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωx_BL = 1/ε²*ωx0_B0 .+ 1/ε*ωx_B1
    # ωy_BL = 1/ε²*ωy0_B0 .+ 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ χx0_B0 .+ ε*(χx_I1 .+ χx_B1)
    # χy_BL = χy_I0 .+ χy0_B0 .+ ε*(χy_I1 .+ χy_B1)

    # buoyancy
    ωx_I0 = -bx/y
    ωy_I0 = -by/y
    χx_I0 = @. (z^3 - z)/6 # bx = z
    χy_I0 = @. (z^2 + z)/2 # by = 1
    c1 = -ωx_I0[nz]
    c2 = ωy_I0[nz]
    ωx0_B0 = @. exp(q*z_s)*(c1*cos(q*z_s) + c2*sin(q*z_s))
    ωy0_B0 = @. exp(q*z_s)*(c1*sin(q*z_s) - c2*cos(q*z_s))
    χx0_B2 = @. exp(q*z_s)*(c2*cos(q*z_s) - c1*sin(q*z_s))/(2q^2)
    χy0_B2 = @. exp(q*z_s)*(c1*sin(q*z_s) - c2*cos(q*z_s))/(2q^2)
    c1 = -5q/6 # bx = z
    c2 = q/6 # by = 1
    χx_I1 = @. -c2*z/(2*H*q^2)
    χy_I1 = @. +c1*z/(2*H*q^2)
    ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    χx_B1 = @. exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))/(2q^2)
    χy_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))/(2q^2)
    ωx_BL = ωx_I0 .+ ωx0_B0 .+ 1/ε*ωx_B1
    ωy_BL = ωy_I0 .+ ωy0_B0 .+ 1/ε*ωy_B1
    χx_BL = χx_I0 .+ ε*(χx_I1 .+ χx_B1) .+ ε²*χx0_B2
    χy_BL = χy_I0 .+ ε*(χy_I1 .+ χy_B1) .+ ε²*χy0_B2

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
    ax[2, 1].set_xlim(-2/ε, 2/ε)
    ax[2, 1].set_ylim(-1, -1 + 10*ε/q)
    ax[2, 2].set_xlim(-2*ε, 2*ε)
    ax[2, 2].set_ylim(-1, -1 + 10*ε/q)
    ax[1, 2].set_yticklabels([])
    ax[2, 2].set_yticklabels([])
    savefig("scratch/images/omega_chi.png")
    println("scratch/images/omega_chi.png")
    plt.close()
end

# test_1d()

### 