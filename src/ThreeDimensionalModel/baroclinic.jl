"""
    A = get_baroclinic_LHS(z, ε², f)

Create LU-factored matrix for 1D baroclinc problem:
    -ε²∂zz(ωˣ) - ωʸ =  ∂y(b),
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
      -∂zz(χˣ) - ωˣ = 0,
      -∂zz(χʸ) - ωʸ = 0,
with bc
    z = 0:   ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    z = -H:  χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
"""
function get_baroclinic_LHS(z, ε², f)
    # create 1D grid
    nz = size(z, 1)
    p = reshape(z, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nz])
    g = Grid(1, p, t, e)

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
    for k=1:g.nt
        # stiffness and mass matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

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

    # z = 0: ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    push!(A, (ωxmap[nz], ωxmap[nz], 1))
    push!(A, (ωymap[nz], ωymap[nz], 1))
    push!(A, (χxmap[nz], χxmap[nz], 1))
    push!(A, (χymap[nz], χymap[nz], 1))

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    return lu(A)
end

"""
    r = get_baroclinic_RHS(z, bx, by, τx, τy, Ux, Uy, ε²)

Create RHS vector for 1D baroclinc problem:
    -ε²∂zz(ωˣ) - ωʸ =  ∂y(b),
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
      -∂zz(χˣ) - ωˣ = 0,
      -∂zz(χʸ) - ωʸ = 0,
with bc
    z = 0:   ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    z = -H:  χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
"""
function get_baroclinic_RHS(z, bx, by, τx, τy, Ux, Uy, ε²)
    # create 1D grid
    nz = size(z, 1)
    p = reshape(z, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nz])
    g = Grid(1, p, t, e)

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
    r = zeros(N)
    for k=1:g.nt
        # mass matrix
        M = J.dets[k]*s.M

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
    end

    # z = -H: χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
    r[ωxmap[1]] = 0
    r[ωymap[1]] = 0

    # z = 0: ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    r[ωxmap[nz]] = -τy/ε²
    r[ωymap[nz]] = τx/ε²
    r[χxmap[nz]] = Uy
    r[χymap[nz]] = -Ux

    return r
end

function get_transport_ω_and_χ(baroclinic_LHSs, g_sfc, g_cols, z_cols, H, ε²; showplots=false)
    # pre-allocate 
    ωx_Ux = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    ωy_Ux = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χx_Ux = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χy_Ux = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    
    # this loop is a bit redundant since each node may be shared by a few triangles, 
    # but we only have to do this once per simulation
    for k=1:g_sfc.nt
        n = 0
        for i=1:3
            ig = g_sfc.t[k, i]
            nz = size(z_cols[ig], 1)
            if nz == 1
                n += nz
                continue
            end
            r = get_baroclinic_RHS(z_cols[ig], zeros(nz-1), zeros(nz-1), H[ig]^2, 0, 0, 0, ε²)
            sol = baroclinic_LHSs[ig]\r
            ωx_Ux[k][n+1:n+nz] = sol[0*nz+1:1*nz]
            ωy_Ux[k][n+1:n+nz] = sol[1*nz+1:2*nz]
            χx_Ux[k][n+1:n+nz] = sol[2*nz+1:3*nz]
            χy_Ux[k][n+1:n+nz] = sol[3*nz+1:4*nz]
            n += nz
        end
    end

    if showplots
        ωx_Ux_bot = DGField([ωx_Ux[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3], g_sfc)
        ωy_Ux_bot = DGField([ωy_Ux[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3], g_sfc)
        quick_plot(ωx_Ux_bot, L"\omega^x_{U^x}(-H)", "$out_folder/omegax_Ux_bot.png")
        quick_plot(ωy_Ux_bot, L"\omega^y_{U^x}(-H)}", "$out_folder/omegay_Ux_bot.png")
        # write_vtk(g, "output/baroclinic_Ux.vtu", Dict("ωx_Ux"=>ωx_Ux, "ωy_Ux"=>ωy_Ux, "χx_Ux"=>χx_Ux, "χy_Ux"=>χy_Ux))
    end

    return ωx_Ux, ωy_Ux, χx_Ux, χy_Ux
end

function get_wind_ω_and_χ(baroclinic_LHSs, g_sfc, z_cols, H, ε²; showplots=false)
    # pre-allocate 
    ωx_τx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    ωy_τx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χx_τx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χy_τx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    
    # this loop is a bit redundant since each node may be shared by a few triangles, 
    # but we only have to do this once per simulation
    for k=1:g_sfc.nt
        n = 0
        for i=1:3
            ig = g_sfc.t[k, i]
            nz = size(z_cols[ig], 1)
            if nz == 1
                n += nz
                continue
            end
            r = get_baroclinic_RHS(z_cols[ig], zeros(nz-1), zeros(nz-1), 0, 0, H[ig]^2, 0, ε²)
            sol = baroclinic_LHSs[ig]\r
            ωx_τx[k][n+1:n+nz] = sol[0*nz+1:1*nz]
            ωy_τx[k][n+1:n+nz] = sol[1*nz+1:2*nz]
            χx_τx[k][n+1:n+nz] = sol[2*nz+1:3*nz]
            χy_τx[k][n+1:n+nz] = sol[3*nz+1:4*nz]
            n += nz
        end
    end

    if showplots
        ωx_τx_bot = DGField([ωx_τx[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3], g_sfc)
        ωy_τx_bot = DGField([ωy_τx[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3], g_sfc)
        quick_plot(ωx_τx_bot, L"\omega^x_{\tau^x}(-H)", "$out_folder/omegax_taux_bot.png")
        quick_plot(ωy_τx_bot, L"\omega^y_{\tau^x}(-H)}", "$out_folder/omegay_taux_bot.png")
        # write_vtk(g, "output/baroclinic_taux.vtu", Dict("ωx_τx"=>ωx_τx, "ωy_τx"=>ωy_τx, "χx_τx"=>χx_τx, "χy_τx"=>χy_τx))
    end

    return ωx_τx, ωy_τx, χx_τx, χy_τx
end

function get_buoyancy_ω_and_χ(m::ModelSetup3D, b; showplots=false)
    # unpack
    g_sfc = m.g_sfc
    g_cols = m.g_cols
    b_cols = m.b_cols
    z_cols = m.z_cols
    Dxs = m.Dxs
    Dys = m.Dys
    ε² = m.ε²
    baroclinic_LHSs = m.baroclinic_LHSs

    # setup arrays
    bvals = [[b(b_cols[k].p[i, :]) for i=1:b_cols[k].np] for k=1:g_sfc.nt]
    bx = [[Dxs[k][i]*bvals[k] for i=1:3] for k=1:g_sfc.nt]
    by = [[Dys[k][i]*bvals[k] for i=1:3] for k=1:g_sfc.nt]

    # solve 
    ωx_b = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    ωy_b = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χx_b = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    χy_b = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
    for k=1:g_sfc.nt
        n = 0
        for i=1:3
            ig = g_sfc.t[k, i]
            nz = size(z_cols[ig], 1)
            if nz ≤ 2
                n += nz
                continue
            end
            r = get_baroclinic_RHS(z_cols[ig], bx[k][i], by[k][i], 0, 0, 0, 0, ε²)
            sol = baroclinic_LHSs[ig]\r
            ωx_b[k][n+1:n+nz] = sol[0*nz+1:1*nz]
            ωy_b[k][n+1:n+nz] = sol[1*nz+1:2*nz]
            χx_b[k][n+1:n+nz] = sol[2*nz+1:3*nz]
            χy_b[k][n+1:n+nz] = sol[3*nz+1:4*nz]
            n += nz
        end
    end 

    if showplots
        ωx_b_bot = DGField([ωx_b[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3], g_sfc)
        ωy_b_bot = DGField([ωy_b[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3], g_sfc)
        quick_plot(ωx_b_bot, L"\omega^x_b(-H)", "$out_folder/omegax_b_bot.png")
        quick_plot(ωy_b_bot, L"\omega^y_b(-H)", "$out_folder/omegay_b_bot.png")
        # write_vtk(g, "output/baroclinic_b.vtu", Dict("ωx_b"=>ωx_b, "ωy_b"=>ωy_b, "χx_b"=>χx_b, "χy_b"=>χy_b))
    end

    return ωx_b, ωy_b, χx_b, χy_b
end

"""
    Dxs, Dys = get_b_gradient_matrices(b_col, g_col, g_sfc, z_cols, k)    

Compute gradient matrices for element column corresponding to surface triangle `k`.
Stored in arrays such that `Dxs[i]` is and (2*nz[i]-2) × (b_col.np) matrix that gives bx
for node column i when multiplied by b in `b_col`.  
"""
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
        Dx = Tuple{Int64,Int64,Float64}[]
        Dy = Tuple{Int64,Int64,Float64}[]
        for j=1:nz-1
            k_tet = findfirst(k -> n+j ∈ g_col.t[k, :] && n+j+1 ∈ g_col.t[k, :], 1:g_col.nt)
            ξx = g_col.J.Js[k_tet, 1, 1]
            ξy = g_col.J.Js[k_tet, 1, 2]
            ηx = g_col.J.Js[k_tet, 2, 1]
            ηy = g_col.J.Js[k_tet, 2, 2]
            ζx = g_col.J.Js[k_tet, 3, 1]
            ζy = g_col.J.Js[k_tet, 3, 2]
            i1_tet = findfirst(i -> g_col.t[k_tet, i] == n+j, 1:g_col.nn) 
            i2_tet = findfirst(i -> g_col.t[k_tet, i] == n+j+1, 1:g_col.nn)
            for l=1:b_col.nn
                push!(Dx, (2j-1, b_col.t[k_tet, l], Dξ[i1_tet, l]*ξx + Dη[i1_tet, l]*ηx + Dζ[i1_tet, l]*ζx))
                push!(Dx, (2j,   b_col.t[k_tet, l], Dξ[i2_tet, l]*ξx + Dη[i2_tet, l]*ηx + Dζ[i2_tet, l]*ζx))
                push!(Dy, (2j-1, b_col.t[k_tet, l], Dξ[i1_tet, l]*ξy + Dη[i1_tet, l]*ηy + Dζ[i1_tet, l]*ζy))
                push!(Dy, (2j,   b_col.t[k_tet, l], Dξ[i2_tet, l]*ξy + Dη[i2_tet, l]*ηy + Dζ[i2_tet, l]*ζy))
            end
        end
        Dxs[i] = sparse((x -> x[1]).(Dx), (x -> x[2]).(Dx), (x -> x[3]).(Dx), 2nz-2, b_col.np)
        Dys[i] = sparse((x -> x[1]).(Dy), (x -> x[2]).(Dy), (x -> x[3]).(Dy), 2nz-2, b_col.np)
        n += nz
    end

    return Dxs, Dys
end