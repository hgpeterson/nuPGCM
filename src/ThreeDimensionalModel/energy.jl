"""
-∫ bz dx dy dz = -∫ H²bσ dx dy dσ
"""
function potential_energy(m::ModelSetup3D, s::ModelState3D)
    # unpack
    H = m.geom.H
    nσ = m.geom.nσ
    g2 = m.geom.g2
    b = s.b

    # fe fields
    σ = FEField(g2.p[:, 3], g2)
    H_g2 = FEField([H[get_i_sfc(i, nσ)] for i ∈ 1:g2.np], g2)

    # # integrand
    # function f(ξ, k)
    #     k_sfc = get_k_sfc(k, nσ)
    #     return -H(ξ, k_sfc)^2*b(ξ, k)*σ(ξ, k)*g2.J.dets[k]
    # end

    # return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)

    return ∫(-(H_g2[g2.t]*g2.φ_qp).^2 .* (b[g2.t]*g2.φ_qp) .* (σ[g2.t]*g2.φ_qp), g2)
end

"""
1/2 ∫ (uˣ)² + (uʸ)² dx dy dz = 1/2 ∫  [ (Huˣ)² + (Huʸ)² ] / H dx dy dσ
"""
function kinetic_energy(m::ModelSetup3D, s::ModelState3D)
    # unpack
    H = m.geom.H
    nσ = m.geom.nσ
    g1 = m.geom.g1
    g2 = m.geom.g2
    χx = s.χx
    χy = s.χy

    # fe fields
    # χx = FEField(χx)
    # χy = FEField(χy)
    H_g2 = FEField([H[get_i_sfc(i, nσ)] for i ∈ 1:g2.np], g2)
    return ∫(0.5./(H_g2[g2.t]*g2.φ_qp) .* ((sum(χy.values.*g1.∂φ_qp[:, :, 3, :], dims=2)[:, 1, :]).^2 .+ (sum(χx.values.*g1.∂φ_qp[:, :, 3, :], dims=2)[:, 1, :]).^2), g1)

    # # integrand
    # function f(ξ, k)
    #     # ux, uy
    #     Hux = -∂σ(χy, ξ, k)
    #     Huy = +∂σ(χx, ξ, k)
    #     k_sfc = get_k_sfc(k, nσ)
    #     return 0.5*(Hux^2 + Huy^2)/H(ξ, k_sfc)*g1.J.dets[k]
    # end

    # return sum(ref_el_quad(ξ -> f(ξ, k), g1.el) for k=1:g1.nt)
end

"""
∫ uᶻb dx dy dz = ∫ Huᶻb dx dy dσ
"""
function buoyancy_production(m::ModelSetup3D, s::ModelState3D)
    # unpack
    g1 = m.geom.g1
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    Hx = m.geom.Hx
    Hy = m.geom.Hy
    χx = s.χx
    χy = s.χy
    b = s.b

    # fields 
    # σ = FEField(g2.p[:, 3], g2)
    χx = FEField(χx)
    χy = FEField(χy)
    σ_g2 = FEField(g2.p[:, 3], g2)
    H_g2 = FEField([H[get_i_sfc(i, nσ)] for i ∈ 1:g2.np], g2)
    Hx_g1 = DGField([Hx[get_k_sfc(k, nσ), mod1(i, 3)] for k ∈ 1:g1.nt, i ∈ 1:g1.nn], g1)
    Hy_g1 = DGField([Hy[get_k_sfc(k, nσ), mod1(i, 3)] for k ∈ 1:g1.nt, i ∈ 1:g1.nn], g1)
    σ = σ_g2[g2.t]*g2.φ_qp
    H = H_g2[g2.t]*g2.φ_qp
    Hx = Hx_g1.values*g1.φ_qp
    Hy = Hy_g1.values*g1.φ_qp
    # Hux = -sum(χy[g1.t].*g1.∂φ_qp[:, :, 3, :], dims=2)[:, 1, :]
    # Huy = +sum(χx[g1.t].*g1.∂φ_qp[:, :, 3, :], dims=2)[:, 1, :]
    # Huσ = sum(χy[g1.t].*g1.∂φ_qp[:, :, 1, :] - χx[g1.t].*g1.∂φ_qp[:, :, 2, :], dims=2)[:, 1, :]
    Hux = zeros(g1.nt, length(g1.el.quad_wts))
    Huy = zeros(g1.nt, length(g1.el.quad_wts))
    Huσ = zeros(g1.nt, length(g1.el.quad_wts))
    for k ∈ 1:g1.nt, i_quad ∈ eachindex(g1.el.quad_wts), i ∈ 1:g1.nn
        Hux[k, i_quad] -= χy[g1.t[k, i]]*g1.∂φ_qp[k, i, 3, i_quad]
        Huy[k, i_quad] += χx[g1.t[k, i]]*g1.∂φ_qp[k, i, 3, i_quad]
        Huσ[k, i_quad] += χy[g1.t[k, i]]*g1.∂φ_qp[k, i, 1, i_quad] - χx[g1.t[k, i]]*g1.∂φ_qp[k, i, 2, i_quad]
    end
    Huz = @. H*Huσ + σ*(Hx*Hux + Hy*Huy)
    b = b[g2.t]*g2.φ_qp
    # Huz = (H_g2[g2.t]*g2.φ_qp).*sum(χy[g1.t].*g1.∂φ_qp[:, :, 1, :] - χx[g1.t].*g1.∂φ_qp[:, :, 2, :], dims=2)[:, 1, :] .+ 
    #       (σ_g2[g2.t]*g2.φ_qp).*(-(Hx_g1.values*g1.φ_qp).*sum(χy[g1.t].*g1.∂φ_qp[:, :, 3, :], dims=2)[:, 1, :] .+
    #                               (Hy_g1.values*g1.φ_qp).*sum(χx[g1.t].*g1.∂φ_qp[:, :, 3, :], dims=2)[:, 1, :])
    return ∫(Huz.*b, g2)

    # # integrand
    # function f(ξ, k)
    #     # ux, uy, uσ
    #     Hux = -∂σ(χy, ξ, k)
    #     Huy = +∂σ(χx, ξ, k)
    #     Huσ = ∂ξ(χy, ξ, k) - ∂η(χx, ξ, k)

    #     # uz
    #     k_sfc = get_k_sfc(k, nσ)
    #     Huz = H(ξ, k_sfc)*Huσ + σ(ξ, k)*Hx(ξ, k_sfc)*Hux + σ(ξ, k)*Hy(ξ, k_sfc)*Huy
    #     return Huz*b(ξ, k)*g2.J.dets[k]
    # end

    # return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end

"""
ε² ∫ ν [ ∂z(ux)^2 + ∂z(uy)^2] dx dy dz = ε² ∫ Hν [ ωx^2 + ωy^2 ] dx dy dσ
"""
function KE_dissipation(m::ModelSetup3D, s::ModelState3D)
    # unpack
    ε² = m.params.ε²
    g1 = m.geom.g1
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    ν = m.forcing.ν
    ωx = s.ωx
    ωy = s.ωy

    ωx = FEField(ωx)
    ωy = FEField(ωy)
    H_g2 = FEField([H[get_i_sfc(i, nσ)] for i ∈ 1:g2.np], g2)

    return ∫(ε²*(ν[g2.t]*g2.φ_qp) .* (H_g2[g2.t]*g2.φ_qp) .* ( (ωx[g1.t]*g1.φ_qp).^2 .+ (ωy[g1.t]*g1.φ_qp).^2 ), g2)

    # # integrand
    # function f(ξ, k)
    #     k_sfc = get_k_sfc(k, nσ)
    #     return ε²*ν(ξ, k)*H(ξ, k_sfc)*(ωx(ξ, k)^2 + ωy(ξ, k)^2)*g2.J.dets[k]
    # end

    # return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
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

    bσ = zeros(g2.nt, length(g2.el.quad_wts))
    for k ∈ 1:g2.nt, i_quad ∈ eachindex(g2.el.quad_wts), i ∈ 1:g2.nn
        bσ[k, i_quad] += b[g2.t[k, i]]*g2.∂φ_qp[k, i, 3, i_quad]
    end
    return ∫(ε²/μ/ϱ*(κ[g2.t]*g2.φ_qp) .* bσ, g2)

    # # integrand
    # function f(ξ, k)
    #     return ε²/μ/ϱ*κ(ξ, k)*∂σ(b, ξ, k)*g2.J.dets[k]
    # end

    # return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end