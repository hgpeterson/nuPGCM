"""
    A = get_barotropic_LHS(νωx_Ux_bot, νωy_Ux_bot, f, β, H, Hx, Hy, ε²)

Generate LU-factored LHS matrix for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function get_barotropic_LHS(g, νωx_Ux_bot, νωy_Ux_bot, f, β, H, Hx, Hy, ε²)
    # unpack
    # g = νωx_Ux_bot.g
    bdy = g.e["bdy"]
    J = g.J
    el = g.el

    # νωx_Ux_bot = νωx_Ux_bot/FEField(H[1:g.np], g)^3
    # νωy_Ux_bot = νωy_Ux_bot/FEField(H[1:g.np], g)^3

    # indices
    N = g.np

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
        T(ξ) = transform_from_ref_el(el, ξ, g.p[g.t[k, 1:3], :])

        # K
        function func_K(ξ, i, j)
            x = T(ξ)
            φx_i = φξ(el, ξ, i)*ξx + φη(el, ξ, i)*ηx
            φy_i = φξ(el, ξ, i)*ξy + φη(el, ξ, i)*ηy
            φx_j = φξ(el, ξ, j)*ξx + φη(el, ξ, j)*ηx
            φy_j = φξ(el, ξ, j)*ξy + φη(el, ξ, j)*ηy
            φ_i = φ(g.el, ξ, i)
            # return -ε²*νωy_Ux_bot(x, k)*(φx_i*φx_j + φy_i*φy_j)*∂x∂ξ
            # return -ε²*νωy_Ux_bot(x, k)*(φx_i*φx_j + φy_i*φy_j)/H(x, k)^3*∂x∂ξ
            return -ε²*3*νωy_Ux_bot(x, k)/H(x, k)*(φx_j*Hx(x, k) + φy_j*Hy(x, k))*φ_i*∂x∂ξ +
                   -ε²*νωy_Ux_bot(x, k)*(φx_i*φx_j + φy_i*φy_j)*∂x∂ξ
            # return -ε²*νωy_Ux_bot(x, k)*(φx_i*φx_j + φy_i*φy_j)*∂x∂ξ
        end
        K = [ref_el_quad(ξ -> func_K(ξ, i, j), el) for i=1:el.n, j=1:el.n]

        # K′
        function func_K′(ξ, i, j)
            x = T(ξ)
            φx_i = φξ(el, ξ, i)*ξx + φη(el, ξ, i)*ηx
            φy_i = φξ(el, ξ, i)*ξy + φη(el, ξ, i)*ηy
            φx_j = φξ(el, ξ, j)*ξx + φη(el, ξ, j)*ηx
            φy_j = φξ(el, ξ, j)*ξy + φη(el, ξ, j)*ηy
            φ_i = φ(g.el, ξ, i)
            # return -ε²*νωx_Ux_bot(x, k)*(φx_i*φy_j - φy_i*φx_j)*∂x∂ξ
            # return -ε²*νωx_Ux_bot(x, k)*(φx_i*φy_j - φy_i*φx_j)/H(x, k)^3*∂x∂ξ
            return -ε²*3*νωx_Ux_bot(x, k)/H(x, k)*(φy_j*Hx(x, k) - φx_j*Hy(x, k))*φ_i*∂x∂ξ +
                   -ε²*νωx_Ux_bot(x, k)*(φx_i*φy_j - φy_i*φx_j)*∂x∂ξ
            # return -ε²*νωx_Ux_bot(x, k)*(φx_i*φy_j - φy_i*φx_j)*∂x∂ξ
        end
        K′ = [ref_el_quad(ξ -> func_K′(ξ, i, j), el) for i=1:el.n, j=1:el.n]

        # J(f/H, Ψ) term
        function func_C(ξ, i, j)
            x = T(ξ)
            φx_j = φξ(el, ξ, j)*ξx + φη(el, ξ, j)*ηx
            φy_j = φξ(el, ξ, j)*ξy + φη(el, ξ, j)*ηy
            φ_i = φ(g.el, ξ, i)
            # return ((H(x, k)*β - (f + β*x[2])*Hy(x, k))*φx_j + (f + β*x[2])*Hx(x, k)*φy_j)*φ_i/H(x, k)^2*∂x∂ξ
            return ((H(x, k)*β - (f + β*x[2])*Hy(x, k))*φx_j + (f + β*x[2])*Hx(x, k)*φy_j)*φ_i*H(x, k)*∂x∂ξ
        end
        C = [ref_el_quad(ξ -> func_C(ξ, i, j), el) for i=1:el.n, j=1:el.n]

        # interior terms
        for i=1:el.n, j=1:el.n
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
    r = get_barotropic_RHS_τ(g_sfc, H, Hx, Hy, τx, τy, τx_y, τy_x, νωx_τ_bot, νωy_τ_bot, ε²)

Generate wind component of RHS vector for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function get_barotropic_RHS_τ(g, H, Hx, Hy, τx, τy, τx_y, τy_x, νωx_τ_bot, νωy_τ_bot, ε²)
    # unpack
    # g = νωx_τ_bot.g
    bdy = g.e["bdy"]
    J = g.J
    el = g.el

    # indices
    N = g.np

    # stamp
    rhs = zeros(N)
    for k=1:g.nt
        # Jacobian
        ∂x∂ξ = J.dets[k]

        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(el, ξ, g.p[g.t[k, 1:3], :])

        # rhs
        function func_r(ξ, i)
            x = T(ξ)
            τ_curl = (τy_x(x, k) - τx_y(x, k))/H(x, k) - (τy(x, k)*Hx(x, k) - τx(x, k)*Hy(x, k))/H(x, k)^2
            νω_τ_bot_div = (∂x(νωx_τ_bot, x, k) + ∂y(νωy_τ_bot, x, k))/H(x, k) - (νωx_τ_bot(x, k)*Hx(x, k) + νωy_τ_bot(x, k)*Hy(x, k))/H(x, k)^2
            φ_i = φ(el, ξ, i)
            return (τ_curl + ε²*νω_τ_bot_div)*φ_i*∂x∂ξ
        end
        r = [ref_el_quad(ξ -> func_r(ξ, i), el) for i=1:el.n]

        rhs[g.t[k, :]] += r
    end

    # boundary nodes 
    for i ∈ bdy
        rhs[i] = 0
    end

    return rhs
end

"""
    r = get_barotropic_RHS_b(m, b, νωx_b_bot, νωy_b_bot)

Generate buoyancy component of RHS vector for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function get_barotropic_RHS_b(m::ModelSetup3D, b, νωx_b_bot, νωy_b_bot; showplots=false)
    # compute JEBAR
    JEBAR = get_JEBAR(m, b, showplots=showplots)

    # unpack
    ε² = m.ε²
    g = m.g_sfc1
    # g = m.g_sfc2
    bdy = g.e["bdy"]
    el = g.el
    H = m.H
    Hx = m.Hx
    Hy = m.Hy

    # stamp
    rhs = zeros(g.np)
    for k=1:g.nt
        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(el, ξ, g.p[g.t[k, 1:3], :])

        # rhs
        function func_r(ξ, i)
            x = T(ξ)
            # νω_b_bot_div = (∂x(νωx_b_bot, x, k) + ∂y(νωy_b_bot, x, k))/H(x, k) - (νωx_b_bot(x, k)*Hx(x, k) + νωy_b_bot(x, k)*Hy(x, k))/H(x, k)^2
            νω_b_bot_div = (∂x(νωx_b_bot, x, k) + ∂y(νωy_b_bot, x, k))*H(x, k)^2 - (νωx_b_bot(x, k)*Hx(x, k) + νωy_b_bot(x, k)*Hy(x, k))*H(x, k)
            φ_i = φ(el, ξ, i)
            # return (-JEBAR(x, k) + ε²*νω_b_bot_div)*φ_i*g.J.dets[k]
            return (-JEBAR(x, k)*H(x, k)^3 + ε²*νω_b_bot_div)*φ_i*g.J.dets[k]
        end
        r = [ref_el_quad(ξ -> func_r(ξ, i), el) for i=1:el.n]

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
    g_sfc1 = m.g_sfc1
    σ = m.σ
    Dxs = m.Dxs
    Dys = m.Dys
    Hx = m.Hx
    Hy = m.Hy

    # compute b gradients
    bx = [Dxs[k, i]'*b.values for k=1:g_sfc1.nt, i=1:g_sfc1.nn]
    by = [Dys[k, i]'*b.values for k=1:g_sfc1.nt, i=1:g_sfc1.nn]

    # compute and store
    JEBAR = zeros(g_sfc1.nt, g_sfc1.nn)
    for k=1:g_sfc1.nt
        for i=1:g_sfc1.nn
            γx = integrate_γ(bx[k, i], σ) 
            γy = integrate_γ(by[k, i], σ) 
            JEBAR[k, i] = Hy[k, i]*γx - Hx[k, i]*γy
        end
    end
    JEBAR = DGField(JEBAR, g_sfc1)

    if showplots
        quick_plot(JEBAR, L"J(1/H, \gamma)", "$out_folder/JEBAR.png")
    end
    return JEBAR
end

"""
    ∫ σf dσ = integrate_γ(f, σ)

Integrate `σ` times DG field `f` over σ using trapezoidal rule.
"""
function integrate_γ(f, σ)
    return sum((f[2k-1]*σ[k] + f[2k]*σ[k+1])/2 * (σ[k+1] - σ[k]) for k=1:length(σ)-1)
end