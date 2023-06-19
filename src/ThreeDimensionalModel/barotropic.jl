"""
    A = get_barotropic_LHS(g_sfc, r_sym, r_asym, f, fy, H, Hx, Hy, ε²)

Generate LU-factored LHS matrix for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function get_barotropic_LHS(g, r_sym, r_asym, f, fy, H, Hx, Hy, ε²)
    # indices
    N = g.np

    # unpack
    bdy = g.e["bdy"]
    J = g.J

    # integration
    quad_wts, quad_pts = quad_weights_points(deg=7, dim=2)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    print("Building barotropic LHS matrix")
    t₀ = time()
    for k=1:g.nt
        if mod(k, Int64(round(0.25*g.nt))) == 0
            print(".")
        end
        # Jacobian terms
        ξx = J.Js[k, 1, 1]
        ξy = J.Js[k, 1, 2]
        ηx = J.Js[k, 2, 1]
        ηy = J.Js[k, 2, 2]
        ∂x∂ξ = J.dets[k]

        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(ξ, g.p[g.t[k, 1:3], :])

        # K
        function func_K(ξ, i, j)
            x = T(ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_sym(x, k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*∂x∂ξ
        end
        K = [ref_el_quad(ξ -> func_K(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # K′
        function func_K′(ξ, i, j)
            x = T(ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_asym(x, k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*∂x∂ξ
        end
        K′ = [ref_el_quad(ξ -> func_K′(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # J(f/H, Ψ) term
        function func_C(ξ, i, j)
            x = T(ξ)
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            φi = φ(g.sf, i, ξ)
            return ((H(x, k)*fy(x, k) - f(x, k)*Hy(x, k))*∂xφ_j + f(x, k)*Hx(x, k)*∂yφ_j)*φi/H(x, k)^2*∂x∂ξ
        end
        C = [ref_el_quad(ξ -> func_C(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # interior terms
        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ∉ bdy 
                push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], K′[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], C[i, j]))
            end
        end
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    println(@sprintf(" (%.1f s)", time() - t₀))

    return lu(A)
end

"""
    r = get_barotropic_RHS_τ(g_sfc, H, Hx, Hy, τx, τy, τx_y, τy_x, ωx_τ_bot, ωy_τ_bot, ε²)

Generate wind component of RHS vector for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function get_barotropic_RHS_τ(g, H, Hx, Hy, τx, τy, τx_y, τy_x, ωx_τ_bot, ωy_τ_bot, ε²)
    # indices
    N = g.np

    # unpack
    bdy = g.e["bdy"]
    J = g.J

    # integration
    quad_wts, quad_pts = quad_weights_points(deg=7, dim=2)

    # stamp
    rhs = zeros(N)
    for k=1:g.nt
        # Jacobian
        ∂x∂ξ = J.dets[k]

        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(ξ, g.p[g.t[k, 1:3], :])

        # rhs
        function func_r(ξ, i)
            x = T(ξ)
            τ_curl = (τy_x(x, k) - τx_y(x, k))/H(x, k) - (τy(x, k)*Hx(x, k) - τx(x, k)*Hy(x, k))/H(x, k)^2
            ω_τ_bot_div = ∂x(ωx_τ_bot, x, k) + ∂y(ωy_τ_bot, x, k)
            φi = φ(g.sf, i, ξ)
            return (τ_curl + ε²*ω_τ_bot_div)*φi*∂x∂ξ
        end
        r = [ref_el_quad(ξ -> func_r(ξ, i), quad_wts, quad_pts) for i=1:g.nn]

        rhs[g.t[k, :]] += r
    end

    # boundary nodes 
    for i ∈ bdy
        rhs[i] = 0
    end

    return rhs
end

"""
    r = get_barotropic_RHS_b(m, b, ωx_b_bot, ωy_b_bot)

Generate wind component of RHS vector for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function get_barotropic_RHS_b(m::ModelSetup3D, b, ωx_b_bot, ωy_b_bot)
    # compute JEBAR
    JEBAR = get_JEBAR(m, b)

    # unpack
    g = m.g_sfc
    H = m.H 
    Hx = m.Hx 
    Hy = m.Hy
    ε² = m.ε²

    # indices
    N = g.np

    # unpack
    bdy = g.e["bdy"]
    J = g.J

    # integration
    quad_wts, quad_pts = quad_weights_points(deg=7, dim=2)

    # stamp
    rhs = zeros(N)
    for k=1:g.nt
        # Jacobian
        ∂x∂ξ = J.dets[k]

        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(ξ, g.p[g.t[k, 1:3], :])

        # rhs
        function func_r(ξ, i)
            x = T(ξ)
            ω_b_bot_div = ∂x(ωx_b_bot, x, k) + ∂y(ωy_b_bot, x, k)
            φi = φ(g.sf, i, ξ)
            return (-JEBAR(x, k) + ε²*ω_b_bot_div)*φi*∂x∂ξ
        end
        r = [ref_el_quad(ξ -> func_r(ξ, i), quad_wts, quad_pts) for i=1:g.nn]

        rhs[g.t[k, :]] += r
    end

    # boundary nodes 
    for i ∈ bdy
        rhs[i] = 0
    end

    return rhs
end

function get_JEBAR(m::ModelSetup3D, b; showplots=false)
    # unpack
    g_sfc = m.g_sfc
    p_to_tri = m.p_to_tri
    z_cols = m.z_cols
    nzs = m.nzs
    Dxs = m.Dxs
    Dys = m.Dys
    H = m.H
    Hx = m.Hx
    Hy = m.Hy

    # compute b gradients
    bx = [Dxs[k][i]*b[k].values for k=1:g_sfc.nt, i=1:g_sfc.nn]
    by = [Dys[k][i]*b[k].values for k=1:g_sfc.nt, i=1:g_sfc.nn]

    # compute and store
    JEBAR = zeros(g_sfc.nt, g_sfc.nn)
    for i=1:g_sfc.np
        # keep coastline set to zero
        nz = nzs[i]
        if nz == 1
            continue
        end

        # create 1D grid
        p = reshape(z_cols[i], (nz, 1))
        t = [i + j - 1 for i=1:nz-1, j=1:2]
        e = Dict("bot"=>[1], "sfc"=>[nz])
        g = Grid(1, p, t, e)

        # compute JEBAR with bx and by from each different element column
        for I ∈ p_to_tri[i]
            γx = integrate_γ(g, bx[I]) 
            γy = integrate_γ(g, by[I]) 
            JEBAR[I] = (Hx[i]*γy - Hy[i]*γx)/H[i]^2
        end
    end
    JEBAR = DGField(JEBAR, g_sfc)

    if showplots
        quick_plot(JEBAR*H^2, L"-H^2 J(1/H, \gamma)", "$out_folder/JEBAR.png")
    end
    return JEBAR
end

"""
    ∫ zf dz = integrate_γ(g, f)

Integrat z*f over [-H, 0] for DG array `f` over grid `g` using trapezoidal rule.
"""
function integrate_γ(g, f)
    return sum((f[2k-1]*g.p[g.t[k, 2]] + f[2k]*g.p[g.t[k, 1]])/2 * (g.p[g.t[k, 2]] - g.p[g.t[k, 1]]) for k=1:g.nt)
end