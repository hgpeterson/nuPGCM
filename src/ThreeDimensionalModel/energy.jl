"""
-∫ bz dx dy dz = -∫ H²bσ dx dy dσ
"""
function potential_energy(m::ModelSetup3D, s::ModelState3D)
    # unpack
    H = m.geom.H
    nσ = m.geom.nσ
    g2 = m.geom.g2
    b = s.b

    # sigma fe field
    σ = FEField(g2.p[:, 3], g2)

    # integrand
    function f(ξ, k)
        k_sfc = get_k_sfc(k, nσ)
        return -H(ξ, k_sfc)^2*b(ξ, k)*σ(ξ, k)*g2.J.dets[k]
    end

    return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end

"""
∫ uᶻb dx dy dz = ∫ Huᶻb dx dy dσ
"""
function buoyancy_production(m::ModelSetup3D, s::ModelState3D)
    # unpack
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    Hx = m.geom.Hx
    Hy = m.geom.Hy
    χx = s.χx
    χy = s.χy
    b = s.b

    # # H*uz = H*H*uσ + σ*Hx*H*ux + σ*Hy*H*uy
    # Hux = [-∂z(χy, [0, 0, 0], k) for k=1:g2.nt]
    # Huy = [+∂z(χx, [0, 0, 0], k) for k=1:g2.nt]
    # Huσ = [∂x(χy, [0, 0, 0], k) - ∂y(χx, [0, 0, 0], k) for k=1:g2.nt]
    # uzb = zeros(g2.nt, g2.nn)
    # for k=1:g2.nt, i=1:g2.nn
    #     ig = g2.t[k, i]
    #     k_sfc = get_k_sfc(k, nσ)
    #     ig_sfc = get_i_sfc(ig, nσ)
    #     i_sfc = mod1(i, 3)
    #     j = k - (k_sfc - 1)*(nσ - 1)
    #     Huz = H[ig_sfc]*Huσ[k] + σ[j]*Hx[k_sfc, i_sfc]*Hux[k] + σ[j]*Hy[k_sfc, i_sfc]*Huy[k]
    #     uzb[k, i] = Huz*b.values[g2.t[k, i]]
    # end

    # # element mass matrix
    # M = mass_matrix(g2.el)

    # # integrate
    # return sum(M*uzb'*g2.J.dets)

    # fe field for σ
    σ = FEField(g2.p[:, 3], g2)

    # fe fields for flow
    χx = FEField(χx)
    χy = FEField(χy)

    # integrand
    function f(ξ, k)
        # ux, uy, uσ
        Hux = -∂z(χy, ξ, k)
        Huy = +∂z(χx, ξ, k)
        Huσ = ∂x(χy, ξ, k) - ∂y(χx, ξ, k)

        # uz
        k_sfc = get_k_sfc(k, nσ)
        Huz = H(ξ, k_sfc)*Huσ + σ(ξ, k)*Hx(ξ, k_sfc)*Hux + σ(ξ, k)*Hy(ξ, k_sfc)*Huy
        return Huz*b(ξ, k)*g2.J.dets[k]
    end

    return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end

"""
ε² ∫ ν [ ∂z(ux)^2 + ∂z(uy)^2] dx dy dz = ε² ∫ Hν [ ωx^2 + ωy^2 ] dx dy dσ
"""
function KE_dissipation(m::ModelSetup3D, s::ModelState3D)
    # unpack
    ε² = m.params.ε²
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    ν = m.forcing.ν
    ωx = s.ωx
    ωy = s.ωy

    # integrand
    function f(ξ, k)
        k_sfc = get_k_sfc(k, nσ)
        return ε²*ν(ξ, k)*H(ξ, k_sfc)*(ωx(ξ, k)^2 + ωy(ξ, k)^2)*g2.J.dets[k]
    end

    return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end

"""
ε²/μϱ ∫ κ ∂z(b) dx dy dz = ε²/μϱ ∫ κ ∂σ(b)  dx dy dσ
"""
function PE_production(m::ModelSetup3D, s::ModelState3D)
    # unpack
    ε² = m.params.ε²
    μ = m.params.μ
    ϱ = m.params.ϱ
    g2 = m.geom.g2
    κ = m.forcing.κ
    b = s.b

    # integrand
    function f(ξ, k)
        return ε²/μ/ϱ*κ(ξ, k)∂z(b, ξ, k)*g2.J.dets[k]
    end

    return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end