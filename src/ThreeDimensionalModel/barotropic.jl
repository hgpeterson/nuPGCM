"""
    A = build_barotropic_LHS(params::Params, geom::Geometry, νωx_Ux_bot, νωy_Ux_bot)

Generate LU-factored LHS matrix for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function build_barotropic_LHS(params::Params, geom::Geometry, νωx_Ux_bot, νωy_Ux_bot)
    # unpack
    f = params.f
    β = params.β
    ε² = params.ε²
    H = geom.H
    Hx = geom.Hx
    Hy = geom.Hy
    g = νωx_Ux_bot.g
    bdy = g.e["bdy"]
    J = g.J
    el = g.el

    # FEField for f
    f_fe = FEField(x->f + β*x[2], g)

    # indices
    N = g.np

    # integrands
    function ∫K(ξ, i, j, k)
        φx_i = φξ(el, ξ, i)*J.Js[k, 1, 1] + φη(el, ξ, i)*J.Js[k, 2, 1]
        φy_i = φξ(el, ξ, i)*J.Js[k, 1, 2] + φη(el, ξ, i)*J.Js[k, 2, 2]
        φx_j = φξ(el, ξ, j)*J.Js[k, 1, 1] + φη(el, ξ, j)*J.Js[k, 2, 1]
        φy_j = φξ(el, ξ, j)*J.Js[k, 1, 2] + φη(el, ξ, j)*J.Js[k, 2, 2]
        φ_i = φ(g.el, ξ, i)
        return -ε²*(3*νωy_Ux_bot(ξ, k)/H(ξ, k)*(φx_j*Hx(ξ, k) + φy_j*Hy(ξ, k))*φ_i +
                      νωy_Ux_bot(ξ, k)*(φx_i*φx_j + φy_i*φy_j))*J.dets[k]
    end
    function ∫K′(ξ, i, j, k)
        φx_i = φξ(el, ξ, i)*J.Js[k, 1, 1] + φη(el, ξ, i)*J.Js[k, 2, 1]
        φy_i = φξ(el, ξ, i)*J.Js[k, 1, 2] + φη(el, ξ, i)*J.Js[k, 2, 2]
        φx_j = φξ(el, ξ, j)*J.Js[k, 1, 1] + φη(el, ξ, j)*J.Js[k, 2, 1]
        φy_j = φξ(el, ξ, j)*J.Js[k, 1, 2] + φη(el, ξ, j)*J.Js[k, 2, 2]
        φ_i = φ(g.el, ξ, i)
        return -ε²*(3*νωx_Ux_bot(ξ, k)/H(ξ, k)*(φy_j*Hx(ξ, k) - φx_j*Hy(ξ, k))*φ_i +
                      νωx_Ux_bot(ξ, k)*(φx_i*φy_j - φy_i*φx_j))*J.dets[k]
    end
    function ∫C(ξ, i, j, k)
        φx_j = φξ(el, ξ, j)*J.Js[k, 1, 1] + φη(el, ξ, j)*J.Js[k, 2, 1]
        φy_j = φξ(el, ξ, j)*J.Js[k, 1, 2] + φη(el, ξ, j)*J.Js[k, 2, 2]
        φ_i = φ(g.el, ξ, i)
        return ((H(ξ, k)*β - f_fe(ξ, k)*Hy(ξ, k))*φx_j + f_fe(ξ, k)*Hx(ξ, k)*φy_j)*φ_i*H(ξ, k)*J.dets[k]
    end

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    @showprogress "Building barotropic LHS matrix..." for k=1:g.nt, i=1:el.n, j=1:el.n
        if g.t[k, i] ∉ bdy 
            push!(A, (g.t[k, i], g.t[k, j], ref_el_quad(ξ -> ∫K(ξ, i, j, k), el) +
                                            ref_el_quad(ξ -> ∫K′(ξ, i, j, k), el) +
                                            ref_el_quad(ξ -> ∫C(ξ, i, j, k), el)))
        end
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return lu(A)
end

"""
    r = build_barotropic_RHS_τ(params::Params, geom::Geometry, forcing::Forcing, νωx_τ_bot, νωy_τ_bot)

Generate wind component of RHS vector for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function build_barotropic_RHS_τ(params::Params, geom::Geometry, forcing::Forcing, νωx_τ_bot, νωy_τ_bot)
    # unpack
    ε² = params.ε²
    H = geom.H
    Hx = geom.Hx
    Hy = geom.Hy
    τx = forcing.τx
    τy = forcing.τy
    τx_y = forcing.τx_y
    τy_x = forcing.τy_x
    g = νωx_τ_bot.g
    bdy = g.e["bdy"]
    J = g.J
    el = g.el
    N = g.np

    # stamp
    rhs = zeros(N)
    for k=1:g.nt
        # rhs
        function func_r(ξ, i)
            τ_curl = (τy_x(ξ, k) - τx_y(ξ, k))*H(ξ, k)^2 - (τy(ξ, k)*Hx(ξ, k) - τx(ξ, k)*Hy(ξ, k))*H(ξ, k)
            νω_τ_bot_div = (∂ξ(νωx_τ_bot, ξ, k) + ∂η(νωy_τ_bot, ξ, k))*H(ξ, k)^2 - (νωx_τ_bot(ξ, k)*Hx(ξ, k) + νωy_τ_bot(ξ, k)*Hy(ξ, k))*H(ξ, k)
            φ_i = φ(el, ξ, i)
            return (τ_curl + ε²*νω_τ_bot_div)*φ_i*J.dets[k]
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
    r = build_barotropic_RHS_b(m::ModelSetup3D, b, νωx_b_bot, νωy_b_bot; showplots=false)

Generate buoyancy component of RHS vector for the problem
    ε²[ ∂x(r_sym ∂x(Ψ)) + ∂y(r_sym ∂y(Ψ)) + ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) ] - J(f/H, Ψ)
        = -J(1/H, γ) + z⋅(∇×τ/H) + ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function build_barotropic_RHS_b(m::ModelSetup3D, b, νωx_b_bot, νωy_b_bot; showplots=false)
    # compute JEBAR
    JEBAR = build_JEBAR(m, b, showplots=showplots)

    # unpack
    ε² = m.params.ε²
    g = m.geom.g_sfc1
    bdy = g.e["bdy"]
    el = g.el
    H = m.geom.H
    Hx = m.geom.Hx
    Hy = m.geom.Hy

    # stamp
    rhs = zeros(g.np)
    for k=1:g.nt
        # rhs
        function func_r(ξ, i)
            νω_b_bot_div = (∂ξ(νωx_b_bot, ξ, k) + ∂η(νωy_b_bot, ξ, k))*H(ξ, k)^2 - (νωx_b_bot(ξ, k)*Hx(ξ, k) + νωy_b_bot(ξ, k)*Hy(ξ, k))*H(ξ, k)
            φ_i = φ(el, ξ, i)
            return (-JEBAR(ξ, k)*H(ξ, k)^3 + ε²*νω_b_bot_div)*φ_i*g.J.dets[k]
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
    JEBAR = build_JEBAR(m::ModelSetup3D, b; showplots=false)
"""
function build_JEBAR(m::ModelSetup3D, b; showplots=false)
    # unpack
    g_sfc1 = m.geom.g_sfc1
    σ = m.geom.σ
    Hx = m.geom.Hx
    Hy = m.geom.Hy
    Dx = m.inversion.Dx
    Dy = m.inversion.Dy

    # compute gradients
    bx = reshape(Dx*b.values, (g_sfc1.nt, g_sfc1.nn, :))
    by = reshape(Dy*b.values, (g_sfc1.nt, g_sfc1.nn, :))

    # compute and store
    JEBAR = zeros(g_sfc1.nt, g_sfc1.nn)
    for k=1:g_sfc1.nt
        for i=1:g_sfc1.nn
            γx = integrate_γ(bx[k, i, :], σ) 
            γy = integrate_γ(by[k, i, :], σ) 
            JEBAR[k, i] = Hy[k, i]*γx - Hx[k, i]*γy
        end
    end
    JEBAR = DGField(JEBAR, g_sfc1)

    if showplots
        f = DGField(m.geom.H[g_sfc1.t].^3 .* JEBAR.values, g_sfc1)
        quick_plot(f, cb_label=L"H^3 J(1/H, \gamma)", filename="$out_folder/JEBAR.png")
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