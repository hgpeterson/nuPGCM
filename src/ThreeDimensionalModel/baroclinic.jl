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
    bot = 1
    sfc = nz
    e = Dict("bot"=>[bot], "sfc"=>[sfc])
    g = Grid(1, p, t, e)

    # indices
    ωxmap = 0*g.np+1:1*g.np
    ωymap = 1*g.np+1:2*g.np
    χxmap = 2*g.np+1:3*g.np
    χymap = 3*g.np+1:4*g.np
    N = 4*g.np

    # stiffness and mass matrix over reference element
    K_el = stiffness_matrix(g.el)[:, :, 1, 1]
    M_el = mass_matrix(g.el)

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    for k=1:g.nt
        # stiffness and mass matrices
        K = K_el*g.J.Js[k, 1, 1]^2*g.J.dets[k]
        M = M_el*g.J.dets[k]

        # indices
        ωxi = ωxmap[g.t[k, :]]
        ωyi = ωymap[g.t[k, :]]
        χxi = χxmap[g.t[k, :]]
        χyi = χymap[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ≠ bot &&  g.t[k, i] ≠ sfc
                # -ε²∂zz(ωx)
                push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
                # -ωy
                push!(A, (ωxi[i], ωyi[j], -f*M[i, j]))

                # -ε²∂zz(ωy)
                push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
                # +ωx
                push!(A, (ωyi[i], ωxi[j], f*M[i, j]))
            end
            if g.t[k, i] ≠ sfc
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
    push!(A, (ωxmap[bot], χxmap[bot], 1))
    push!(A, (ωymap[bot], χymap[bot], 1))

    # z = 0: ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    push!(A, (ωxmap[sfc], ωxmap[sfc], 1))
    push!(A, (ωymap[sfc], ωymap[sfc], 1))
    push!(A, (χxmap[sfc], χxmap[sfc], 1))
    push!(A, (χymap[sfc], χymap[sfc], 1))

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    return lu(A)
end

"""
    r = get_baroclinic_RHS(z, bx, by, Ux, Uy, τx, τy, ε²)

Create RHS vector for 1D baroclinc problem:
    -ε²∂zz(ωˣ) - ωʸ =  ∂y(b),
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
      -∂zz(χˣ) - ωˣ = 0,
      -∂zz(χʸ) - ωʸ = 0,
with bc
    z = 0:   ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    z = -H:  χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
"""
function get_baroclinic_RHS(z, bx, by, Ux, Uy, τx, τy, ε²)
    # create 1D grid
    nz = size(z, 1)
    p = reshape(z, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    bot = 1
    sfc = nz
    e = Dict("bot"=>[bot], "sfc"=>[sfc])
    g = Grid(1, p, t, e)

    # indices
    ωxmap = 0*g.np+1:1*g.np
    ωymap = 1*g.np+1:2*g.np
    χxmap = 2*g.np+1:3*g.np
    χymap = 3*g.np+1:4*g.np
    N = 4*g.np

    # mass matrix over reference element
    M_el = mass_matrix(g.el)

    # stamp system
    r = zeros(N)
    for k=1:g.nt
        # mass matrix
        M = M_el*g.J.dets[k]

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
        else
            error("Unsupported length of buoyancy gradient vector for baroclinc problem. Expected $(g.nt), $(2g.nt), or $(g.np), got $(length(bx)).")
        end
    end

    # z = -H: χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
    r[ωxmap[bot]] = 0
    r[ωymap[bot]] = 0

    # z = 0: ωˣ = -τʸ/ε², ωʸ = τˣ/ε², χˣ = Uʸ, χʸ = -Uˣ,
    r[ωxmap[sfc]] = -τy/ε²
    r[ωymap[sfc]] = τx/ε²
    r[χxmap[sfc]] = Uy
    r[χymap[sfc]] = -Ux

    return r
end

function get_transport_ω_and_χ(baroclinic_LHSs, g_sfc1, σ, H, ε²; showplots=false)
    # pre-allocate 
    nσ = length(σ)
    ωx_Ux = zeros(g_sfc1.np, nσ)
    ωy_Ux = zeros(g_sfc1.np, nσ)
    χx_Ux = zeros(g_sfc1.np, nσ)
    χy_Ux = zeros(g_sfc1.np, nσ)
    
    # compute and store
    for i=1:g_sfc1.np
        # H = 0 solution: ωʸ = -3σ, all else zeros
        if i ∈ g_sfc1.e["bdy"]
            ωy_Ux[i, :] = -3*σ
            continue
        end

        # get rhs with Uˣ = H^2 and all else zeros
        r = get_baroclinic_RHS(σ*H[i], zeros(nσ-1), zeros(nσ-1), H[i]^2, 0, 0, 0, ε²)

        # solve baroclinc problem
        sol = baroclinic_LHSs[i]\r

        # store 
        ωx_Ux[i, :] = sol[0*nσ+1:1*nσ]
        ωy_Ux[i, :] = sol[1*nσ+1:2*nσ]
        χx_Ux[i, :] = sol[2*nσ+1:3*nσ]
        χy_Ux[i, :] = sol[3*nσ+1:4*nσ]
    end

    if showplots
        ωx_Ux_bot = FEField([ωx_Ux[i, 1] for i=1:g_sfc1.np], g_sfc1)
        ωy_Ux_bot = FEField([ωy_Ux[i, 1] for i=1:g_sfc1.np], g_sfc1)
        quick_plot(ωx_Ux_bot, L"\omega^x_{U^x}(-H)", "$out_folder/omegax_Ux_bot.png")
        quick_plot(ωy_Ux_bot, L"\omega^y_{U^x}(-H)}", "$out_folder/omegay_Ux_bot.png")
        # write_vtk(g, "output/baroclinic_Ux.vtu", Dict("ωx_Ux"=>ωx_Ux, "ωy_Ux"=>ωy_Ux, "χx_Ux"=>χx_Ux, "χy_Ux"=>χy_Ux))
    end

    return ωx_Ux, ωy_Ux, χx_Ux, χy_Ux
end

function get_wind_ω_and_χ(baroclinic_LHSs, g_sfc1, σ, H, ε²; showplots=false)
    # pre-allocate 
    nσ = length(σ)
    ωx_τx = zeros(g_sfc1.np, nσ)
    ωy_τx = zeros(g_sfc1.np, nσ)
    χx_τx = zeros(g_sfc1.np, nσ)
    χy_τx = zeros(g_sfc1.np, nσ)
    
    # compute and store
    for i=1:g_sfc1.np
        # H = 0 solution: ωʸ = (3σ + 2)/2ε², all else zeros
        if i ∈ g_sfc1.e["bdy"]
            ωy_τx[i, :] = @. (3*σ + 2)/(2ε²)
            continue
        end

        # get rhs with τˣ = 1 and all else zeros
        r = get_baroclinic_RHS(σ*H[i], zeros(nσ-1), zeros(nσ-1), 0, 0, 1, 0, ε²)

        # solve baroclinc problem
        sol = baroclinic_LHSs[i]\r

        # store
        ωx_τx[i, :] = sol[0*nσ+1:1*nσ]
        ωy_τx[i, :] = sol[1*nσ+1:2*nσ]
        χx_τx[i, :] = sol[2*nσ+1:3*nσ]
        χy_τx[i, :] = sol[3*nσ+1:4*nσ]
    end

    if showplots
        ωx_τx_bot = FEField([ωx_τx[i, 1] for i=1:g_sfc1.np], g_sfc1)
        ωy_τx_bot = FEField([ωy_τx[i, 1] for i=1:g_sfc1.np], g_sfc1)
        quick_plot(ωx_τx_bot, L"\omega^x_{\tau^x}(-H)", "$out_folder/omegax_taux_bot.png")
        quick_plot(ωy_τx_bot, L"\omega^y_{\tau^x}(-H)}", "$out_folder/omegay_taux_bot.png")
        # write_vtk(g, "output/baroclinic_taux.vtu", Dict("ωx_τx"=>ωx_τx, "ωy_τx"=>ωy_τx, "χx_τx"=>χx_τx, "χy_τx"=>χy_τx))
    end

    return ωx_τx, ωy_τx, χx_τx, χy_τx
end

function get_buoyancy_ω_and_χ(m::ModelSetup3D, b; showplots=false)
    # unpack
    g_sfc = m.g_sfc
    p_to_tri = m.p_to_tri
    z_cols = m.z_cols
    nzs = m.nzs
    Dxs = m.Dxs
    Dys = m.Dys
    ε² = m.ε²
    baroclinic_LHSs = m.baroclinic_LHSs

    # setup arrays
    bx = [Dxs[k][i]*b[k].values for k=1:g_sfc.nt, i=1:g_sfc.nn]
    by = [Dys[k][i]*b[k].values for k=1:g_sfc.nt, i=1:g_sfc.nn]

    # pre-allocate 
    ωx_b = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    ωy_b = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    χx_b = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]
    χy_b = [zeros(nzs[g_sfc.t[k, i]]) for k=1:g_sfc.nt, i=1:g_sfc.nn]

    # compute and store
    for i=1:g_sfc.np
        # keep coastline set to zero
        nz = nzs[i]
        if nz == 1
            continue
        end

        # solve baroclinic problem with bx and by from each different element column
        for I ∈ p_to_tri[i]
            # get rhs with bx and by and all else zeros
            r = get_baroclinic_RHS(z_cols[i], bx[I], by[I], 0, 0, 0, 0, ε²)

            # solve baroclinc problem
            sol = baroclinic_LHSs[i]\r

            # store
            ωx_b[I] = sol[0*nz+1:1*nz]
            ωy_b[I] = sol[1*nz+1:2*nz]
            χx_b[I] = sol[2*nz+1:3*nz]
            χy_b[I] = sol[3*nz+1:4*nz]
        end
    end

    if showplots
        ωx_b_bot = DGField([ωx_b[k, i][1] for k=1:g_sfc.nt, i=1:g_sfc.nn], g_sfc)
        ωy_b_bot = DGField([ωy_b[k, i][1] for k=1:g_sfc.nt, i=1:g_sfc.nn], g_sfc)
        quick_plot(ωx_b_bot, L"\omega^x_b(-H)", "$out_folder/omegax_b_bot.png")
        quick_plot(ωy_b_bot, L"\omega^y_b(-H)", "$out_folder/omegay_b_bot.png")
        # write_vtk(g, "output/baroclinic_b.vtu", Dict("ωx_b"=>ωx_b, "ωy_b"=>ωy_b, "χx_b"=>χx_b, "χy_b"=>χy_b))
    end

    return ωx_b, ωy_b, χx_b, χy_b
end

"""
    Dxs, Dys = get_b_gradient_matrices(b_col, g_col, nzs)    

Compute gradient matrices for element column `g_col`.
Stored in arrays such that `Dxs[i]` is and (2*nz[i]-2) × (b_col.np) matrix that gives bx
for node column i when multiplied by b in `b_col`.  
"""
function get_b_gradient_matrices(g1, g2, σ, H, Hx, Hy) 
    # unpack
    w1 = g1.el
    w2 = g2.el
    J = g1.J
    nσ = length(σ)
    g_sfc2 = H.g

    Dξ = [φξ(w2, w1.p[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dη = [φη(w2, w1.p[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dζ = [φζ(w2, w1.p[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dxs = Matrix{SparseMatrixCSC}(undef, g_sfc2.nt, 3)
    Dys = Matrix{SparseMatrixCSC}(undef, g_sfc2.nt, 3)
    @showprogress "Computing buoyancy gradient matrices..." for k=1:g_sfc2.nt
        for i=1:3
            i1 = i 
            i2 = i + 3
            Dx = Tuple{Int64,Int64,Float64}[]
            Dy = Tuple{Int64,Int64,Float64}[]
            for j=1:nσ-1
                k_w = (nσ - 1)*(k - 1) + j
                jac = J.Js[k_w, :, :]
                for l=1:w2.n
                    push!(Dx, (2j-1, g2.t[k_w, l], Dξ[i1, l]*jac[1, 1] + Dη[i1, l]*jac[2, 1] + Dζ[i1, l]*jac[3, 1]))
                    push!(Dy, (2j-1, g2.t[k_w, l], Dξ[i1, l]*jac[1, 2] + Dη[i1, l]*jac[2, 2] + Dζ[i1, l]*jac[3, 2]))
                    push!(Dx, (2j-1, g2.t[k_w, l], -σ[j]*Hx[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i1, l]*jac[1, 3] + Dη[i1, l]*jac[2, 3] + Dζ[i1, l]*jac[3, 3])))
                    push!(Dy, (2j-1, g2.t[k_w, l], -σ[j]*Hy[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i1, l]*jac[1, 3] + Dη[i1, l]*jac[2, 3] + Dζ[i1, l]*jac[3, 3])))

                    push!(Dx, (2j, g2.t[k_w, l], Dξ[i2, l]*jac[1, 1] + Dη[i2, l]*jac[2, 1] + Dζ[i2, l]*jac[3, 1]))
                    push!(Dy, (2j, g2.t[k_w, l], Dξ[i2, l]*jac[1, 2] + Dη[i2, l]*jac[2, 2] + Dζ[i2, l]*jac[3, 2]))
                    push!(Dx, (2j, g2.t[k_w, l], -σ[j]*Hx[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i2, l]*jac[1, 3] + Dη[i2, l]*jac[2, 3] + Dζ[i2, l]*jac[3, 3])))
                    push!(Dy, (2j, g2.t[k_w, l], -σ[j]*Hy[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i2, l]*jac[1, 3] + Dη[i2, l]*jac[2, 3] + Dζ[i2, l]*jac[3, 3])))
                end
            end
            Dxs[k, i] = sparse((x -> x[1]).(Dx), (x -> x[2]).(Dx), (x -> x[3]).(Dx), 2nσ-2, g2.np)
            Dys[k, i] = sparse((x -> x[1]).(Dy), (x -> x[2]).(Dy), (x -> x[3]).(Dy), 2nσ-2, g2.np)
        end
    end

    return Dxs, Dys
end