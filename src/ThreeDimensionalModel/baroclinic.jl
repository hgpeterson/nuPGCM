"""
    A = build_baroclinic_LHS(g, ν, H, ε², f)

Create LU-factored matrix for 1D baroclinc problem:
    -ε²∂zz(νωˣ) - fωʸ =  ∂y(b),
    -ε²∂zz(νωʸ) + fωˣ = -∂x(b),
      -∂zz(χˣ) - ωˣ = 0,
      -∂zz(χʸ) - ωʸ = 0,
with bc
    z = 0:   ωˣ = -τʸ/νε², ωʸ = τˣ/νε², χˣ = Uʸ, χʸ = -Uˣ,
    z = -H:  χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
"""
function build_baroclinic_LHS(g::Grid, ν, H, ε², f)
    # unpack
    J = g.J
    el = g.el

    # indices
    ωxmap = 0*g.np+1:1*g.np
    ωymap = 1*g.np+1:2*g.np
    χxmap = 2*g.np+1:3*g.np
    χymap = 3*g.np+1:4*g.np
    N = 4*g.np
    bot = g.e["bot"][1]
    sfc = g.e["sfc"][1]

    # make ν a finite element field
    ν = FEField(ν, g)

    # stiffness and mass matrices on reference element
    K_el = stiffness_matrix(el)[1, 1, :, :]
    M_el = mass_matrix(el)

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    for k=1:g.nt
        # scale by jacobian
        K = K_el*J.Js[k, 1, 1]^2*J.dets[k]
        M = M_el*J.dets[k]

        # ∫ ν ∂φⱼ∂φᵢ + ∫ ∂(ν) φⱼ∂φᵢ
        νK = [ref_el_quad(ξ -> ν(ξ, k)*φξ(el, ξ, i)*φξ(el, ξ, j)*J.Js[k, 1, 1]^2*J.dets[k], el) for i=1:el.n, j=1:el.n] + 
             [ref_el_quad(ξ -> ∂(ν, ξ, k, 1)*φξ(el, ξ, i)*φ(el, ξ, j)*J.Js[k, 1, 1]*J.dets[k], el) for i=1:el.n, j=1:el.n]

        # indices
        ωxi = ωxmap[g.t[k, :]]
        ωyi = ωymap[g.t[k, :]]
        χxi = χxmap[g.t[k, :]]
        χyi = χymap[g.t[k, :]]

        for i=1:el.n, j=1:el.n
            if g.t[k, i] ≠ bot &&  g.t[k, i] ≠ sfc
                # -ε²∂zz(ν*ωx)
                push!(A, (ωxi[i], ωxi[j], ε²/H^2*νK[i, j]))
                # -ωy
                push!(A, (ωxi[i], ωyi[j], -f*M[i, j]))

                # -ε²∂zz(ν*ωy)
                push!(A, (ωyi[i], ωyi[j], ε²/H^2*νK[i, j]))
                # +ωx
                push!(A, (ωyi[i], ωxi[j], f*M[i, j]))
            end
            if g.t[k, i] ≠ sfc
                # -∂zz(χx)
                push!(A, (χxi[i], χxi[j], 1/H^2*K[i, j]))
                # -ωx
                push!(A, (χxi[i], ωxi[j], -M[i, j]))

                # -∂zz(χy)
                push!(A, (χyi[i], χyi[j], 1/H^2*K[i, j]))
                # -ωy
                push!(A, (χyi[i], ωyi[j], -M[i, j]))
            end
        end
    end

    # z = -H: χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
    push!(A, (ωxmap[bot], χxmap[bot], 1))
    push!(A, (ωymap[bot], χymap[bot], 1))

    # z = 0: ν*ε²*ωˣ = -τʸ, ν*ε²*ωʸ = τˣ, χˣ = Uʸ, χʸ = -Uˣ,
    push!(A, (ωxmap[sfc], ωxmap[sfc], ν[sfc]*ε²))
    push!(A, (ωymap[sfc], ωymap[sfc], ν[sfc]*ε²))
    push!(A, (χxmap[sfc], χxmap[sfc], 1))
    push!(A, (χymap[sfc], χymap[sfc], 1))

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    return lu(A)
end

"""
    baroclinic_LHSs = build_baroclinic_LHSs(params::Params, geom::Geometry, forcing::Forcing)

Build baroclinc LHS for each node column on first order grid.
"""
function build_baroclinic_LHSs(params::Params, geom::Geometry, forcing::Forcing)
    g_col = geom.g_col
    ν = forcing.ν
    nσ = geom.nσ
    H = geom.H
    ε² = params.ε²
    f = params.f
    β = params.β
    g_sfc1 = geom.g_sfc1
    in_nodes1 = geom.in_nodes1
    return [build_baroclinic_LHS(g_col, ν[get_col_inds(i, nσ)], H[i], ε², f + β*g_sfc1.p[i, 2]) for i ∈ in_nodes1]
end

"""
    r = build_baroclinic_RHS(g, M_bc, bx, by, Ux, Uy, τx, τy)

Create RHS vector for 1D baroclinc problem:
    -ε²∂zz(νωˣ) - fωʸ =  ∂y(b),
    -ε²∂zz(νωʸ) + fωˣ = -∂x(b),
      -∂zz(χˣ) - ωˣ = 0,
      -∂zz(χʸ) - ωʸ = 0,
with bc
    z = 0:   ωˣ = -τʸ/νε², ωʸ = τˣ/νε², χˣ = Uʸ, χʸ = -Uˣ,
    z = -H:  χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
"""
function build_baroclinic_RHS(g::Grid, M_bc, bx, by, Ux, Uy, τx, τy)
    # indices
    ωxmap = 0*g.np+1:1*g.np
    ωymap = 1*g.np+1:2*g.np
    χxmap = 2*g.np+1:3*g.np
    χymap = 3*g.np+1:4*g.np
    N = 4*g.np
    bot = g.e["bot"][1]
    sfc = g.e["sfc"][1]

    # interior
    r = zeros(N)
    r[ωxmap] = +M_bc*by
    r[ωymap] = -M_bc*bx

    # z = -H: χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0.
    r[ωxmap[bot]] = 0
    r[ωymap[bot]] = 0

    # z = 0: ν*ε²*ωˣ = -τʸ, ν*ε²*ωʸ = τˣ, χˣ = Uʸ, χʸ = -Uˣ,
    r[ωxmap[sfc]] = -τy
    r[ωymap[sfc]] = τx
    r[χxmap[sfc]] = Uy
    r[χymap[sfc]] = -Ux

    return r
end
function build_M_bc(g::Grid)
    # unpack
    J = g.J
    el = g.el

    # mass matrix over reference element
    M_el = mass_matrix(el)

    # stamp system
    M = Tuple{Int64, Int64, Float64}[]
    for k ∈ 1:g.nt, i ∈ 1:g.nn
        push!(M, (g.t[k, i], 2k-1, M_el[i, 1]*J.dets[k]))
        push!(M, (g.t[k, i], 2k,   M_el[i, 2]*J.dets[k]))
    end

    return dropzeros!(sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), g.np, 2*g.nt))
end

function solve_baroclinic_transport(geom::Geometry, baroclinic_LHSs, M_bc; showplots=false)
    # unpack
    g_col = geom.g_col
    nσ = geom.nσ
    in_nodes1 = geom.in_nodes1
    H = geom.H
    g_sfc1 = geom.g_sfc1

    # pre-allocate 
    ωx_Ux = zeros(g_sfc1.np, nσ)
    ωy_Ux = zeros(g_sfc1.np, nσ)
    χx_Ux = zeros(g_sfc1.np, nσ)
    χy_Ux = zeros(g_sfc1.np, nσ)
    
    # compute and store
    for i ∈ eachindex(in_nodes1)
        ig = in_nodes1[i]

        # get rhs with Uˣ = H^2 and all else zeros
        r = build_baroclinic_RHS(g_col, M_bc, zeros(2nσ-2), zeros(2nσ-2), H[ig]^2, 0, 0, 0)

        # solve baroclinc problem
        sol = baroclinic_LHSs[i]\r

        # store 
        ωx_Ux[ig, :] = sol[0*nσ+1:1*nσ]
        ωy_Ux[ig, :] = sol[1*nσ+1:2*nσ]
        χx_Ux[ig, :] = sol[2*nσ+1:3*nσ]
        χy_Ux[ig, :] = sol[3*nσ+1:4*nσ]
    end

    # H = 0 solution: ωʸ = -3σ, all else zeros
    for i ∈ g_sfc1.e["bdy"]
        ωy_Ux[i, :] = -3*g_col.p
    end

    if showplots
        ωx_Ux_bot = FEField(ωx_Ux[:, 1], g_sfc1)
        ωy_Ux_bot = FEField(ωy_Ux[:, 1], g_sfc1)
        quick_plot(ωx_Ux_bot, cb_label=L"\omega^x_{U^x}(-H)",  filename="$out_folder/omegax_Ux_bot.png")
        quick_plot(ωy_Ux_bot, cb_label=L"\omega^y_{U^x}(-H)}", filename="$out_folder/omegay_Ux_bot.png")
    end

    return ωx_Ux, ωy_Ux, χx_Ux, χy_Ux
end

function solve_baroclinic_wind(geom::Geometry, params::Params, baroclinic_LHSs, M_bc; showplots=false)
    # unpack
    g_col = geom.g_col
    nσ = geom.nσ
    in_nodes1 = geom.in_nodes1
    g_sfc1 = geom.g_sfc1
    ε² = params.ε²

    # pre-allocate 
    ωx_τx = zeros(g_sfc1.np, nσ)
    ωy_τx = zeros(g_sfc1.np, nσ)
    χx_τx = zeros(g_sfc1.np, nσ)
    χy_τx = zeros(g_sfc1.np, nσ)
    
    # compute and store
    for i ∈ eachindex(in_nodes1)
        ig = in_nodes1[i]

        # get rhs with τˣ = 1 and all else zeros
        r = build_baroclinic_RHS(g_col, M_bc, zeros(2nσ-2), zeros(2nσ-2), 0, 0, 1, 0)

        # solve baroclinc problem
        sol = baroclinic_LHSs[i]\r

        # store
        ωx_τx[ig, :] = sol[0*nσ+1:1*nσ]
        ωy_τx[ig, :] = sol[1*nσ+1:2*nσ]
        χx_τx[ig, :] = sol[2*nσ+1:3*nσ]
        χy_τx[ig, :] = sol[3*nσ+1:4*nσ]
    end

    # H = 0 solution: ωʸ = (3σ + 2)/2ε², all else zeros
    for i ∈ g_sfc1.e["bdy"]
        ωy_τx[i, :] = @. (3*g_col.p + 2)/(2ε²)
    end

    if showplots
        ωx_τx_bot = FEField(ωx_τx[:, 1], g_sfc1)
        ωy_τx_bot = FEField(ωy_τx[:, 1], g_sfc1)
        quick_plot(ωx_τx_bot, cb_label=L"\omega^x_{\tau^x}(-H)",  filename="$out_folder/omegax_taux_bot.png")
        quick_plot(ωy_τx_bot, cb_label=L"\omega^y_{\tau^x}(-H)}", filename="$out_folder/omegay_taux_bot.png")
    end

    return ωx_τx, ωy_τx, χx_τx, χy_τx
end

function solve_baroclinic_buoyancy(m::ModelSetup3D, b; showplots=false)
    # unpack
    g_sfc1 = m.geom.g_sfc1
    g_col = m.geom.g_col
    nσ = m.geom.nσ
    in_nodes1 = m.geom.in_nodes1
    Dx = m.inversion.Dx
    Dy = m.inversion.Dy
    baroclinic_LHSs = m.inversion.baroclinic_LHSs
    M_bc = m.inversion.M_bc

    # compute gradients
    bx = reshape(Dx*b.values, (g_sfc1.nt, g_sfc1.nn, 2nσ-2))
    by = reshape(Dy*b.values, (g_sfc1.nt, g_sfc1.nn, 2nσ-2))

    # pre-allocate
    ωx_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)
    ωy_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)
    χx_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)
    χy_b = zeros(g_sfc1.nt, g_sfc1.nn, nσ)

    # compute and store
    for i ∈ eachindex(in_nodes1) # H = 0 solution: all zeros
        ig = in_nodes1[i]
        for I ∈ g_sfc1.p_to_t[ig]
            # solve baroclinic problem with bx and by from element column
            r = build_baroclinic_RHS(g_col, M_bc, bx[I, :], by[I, :], 0, 0, 0, 0)
            sol = baroclinic_LHSs[i]\r

            # store
            ωx_b[I, :] = sol[0*nσ+1:1*nσ]
            ωy_b[I, :] = sol[1*nσ+1:2*nσ]
            χx_b[I, :] = sol[2*nσ+1:3*nσ]
            χy_b[I, :] = sol[3*nσ+1:4*nσ]
        end
    end

    if showplots
        ωx_b_bot = DGField(ωx_b[:, :, 1], g_sfc1)
        ωy_b_bot = DGField(ωy_b[:, :, 1], g_sfc1)
        quick_plot(ωx_b_bot, cb_label=L"\omega^x_b(-H)", filename="$out_folder/omegax_b_bot.png")
        quick_plot(ωy_b_bot, cb_label=L"\omega^y_b(-H)", filename="$out_folder/omegay_b_bot.png")
    end

    return ωx_b, ωy_b, χx_b, χy_b
end

"""
    Dx, Dy = build_b_gradient_matrices(geom::Geometry)

Compute gradient matrices for element column in the 3D mesh `g1` (second order `g2`).
`Dx` and `Dy` are (g_sfc.nt)*(g_sfc.nn)*(2*nσ-2) × (g2.np) matrices
that give

    ∂x(b) = ∂ξ(b) - σ*Hx/H ∂σ(b) 
    ∂y(b) = ∂η(b) - σ*Hy/H ∂σ(b) 

for each node column in each surface triangle when multiplied by b. The result should be 
reshaped afterwards, e.g.,

    bx = reshape(Dx*b.values, (g_sfc1.nt, g_sfc1.nn, 2nσ-2))
    by = reshape(Dy*b.values, (g_sfc1.nt, g_sfc1.nn, 2nσ-2))
"""
function build_b_gradient_matrices(geom::Geometry)
    # unpack
    g1 = geom.g1
    g2 = geom.g2
    σ = geom.σ
    nσ = geom.nσ
    g_sfc1 = geom.g_sfc1
    g_sfc2 = geom.g_sfc2
    H = geom.H
    Hx = geom.Hx
    Hy = geom.Hy
    w1 = g1.el
    w2 = g2.el

    Dξ = [φξ(w2, w1.p[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dη = [φη(w2, w1.p[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dζ = [φζ(w2, w1.p[i, :], j) for i=1:w1.n, j=1:w2.n]
    imap = reshape(1:g_sfc1.nt*3*(2nσ-2), (g_sfc1.nt, 3, 2nσ-2))
    Dx = Tuple{Int64,Int64,Float64}[]
    Dy = Tuple{Int64,Int64,Float64}[]
    @showprogress "Building buoyancy gradient matrices..." for k=1:g_sfc1.nt
        for i=1:g_sfc1.nn
            if g_sfc1.t[k, i] ∈ g_sfc1.e["bdy"]  
                continue
            end
            i1 = i 
            i2 = i + 3
            for j=1:nσ-1
                k_w = get_k_w(k, nσ, j)
                jac = g1.J.Js[k_w, :, :]
                for l=1:w2.n
                    push!(Dx, (imap[k, i, 2j-1], g2.t[k_w, l], Dξ[i1, l]*jac[1, 1] + Dη[i1, l]*jac[2, 1] + Dζ[i1, l]*jac[3, 1]))
                    push!(Dy, (imap[k, i, 2j-1], g2.t[k_w, l], Dξ[i1, l]*jac[1, 2] + Dη[i1, l]*jac[2, 2] + Dζ[i1, l]*jac[3, 2]))
                    push!(Dx, (imap[k, i, 2j-1], g2.t[k_w, l], -σ[j]*Hx[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i1, l]*jac[1, 3] + Dη[i1, l]*jac[2, 3] + Dζ[i1, l]*jac[3, 3])))
                    push!(Dy, (imap[k, i, 2j-1], g2.t[k_w, l], -σ[j]*Hy[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i1, l]*jac[1, 3] + Dη[i1, l]*jac[2, 3] + Dζ[i1, l]*jac[3, 3])))

                    push!(Dx, (imap[k, i, 2j], g2.t[k_w, l], Dξ[i2, l]*jac[1, 1] + Dη[i2, l]*jac[2, 1] + Dζ[i2, l]*jac[3, 1]))
                    push!(Dy, (imap[k, i, 2j], g2.t[k_w, l], Dξ[i2, l]*jac[1, 2] + Dη[i2, l]*jac[2, 2] + Dζ[i2, l]*jac[3, 2]))
                    push!(Dx, (imap[k, i, 2j], g2.t[k_w, l], -σ[j+1]*Hx[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i2, l]*jac[1, 3] + Dη[i2, l]*jac[2, 3] + Dζ[i2, l]*jac[3, 3])))
                    push!(Dy, (imap[k, i, 2j], g2.t[k_w, l], -σ[j+1]*Hy[k, i]/H[g_sfc2.t[k, i]]*(Dξ[i2, l]*jac[1, 3] + Dη[i2, l]*jac[2, 3] + Dζ[i2, l]*jac[3, 3])))
                end
            end
        end
    end
    print("Storing buoyancy gradient matrices... ")
    Dx = dropzeros!(sparse((x -> x[1]).(Dx), (x -> x[2]).(Dx), (x -> x[3]).(Dx), g_sfc2.nt*3*(2nσ-2), g2.np))
    Dy = dropzeros!(sparse((x -> x[1]).(Dy), (x -> x[2]).(Dy), (x -> x[3]).(Dy), g_sfc2.nt*3*(2nσ-2), g2.np))
    println("Done.")

    return Dx, Dy
end