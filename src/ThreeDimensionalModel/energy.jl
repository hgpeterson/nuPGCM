"""
∫ uᶻb dx dy dz = ∫ Huᶻb dx dy dσ
"""
function buoyancy_production(m::ModelSetup3D, s::ModelState3D)
    # unpack
    g2 = m.g2
    σ = m.σ
    nσ = m.nσ
    H = m.H
    Hx = m.Hx
    Hy = m.Hy
    χx = s.χx
    χy = s.χy
    b = s.b

    # H*uz = H*H*uσ + σ*Hx*H*ux + σ*Hy*H*uy
    Hux = [-∂z(χy, [0, 0, 0], k) for k=1:g2.nt]
    Huy = [+∂z(χx, [0, 0, 0], k) for k=1:g2.nt]
    Huσ = [∂x(χy, [0, 0, 0], k) - ∂y(χx, [0, 0, 0], k) for k=1:g2.nt]
    println(maximum(abs.(Hux)))
    println(maximum(abs.(Huy)))
    println(maximum(abs.(Huσ)))
    uzb = zeros(g2.nt, g2.nn)
    for k=1:g2.nt, i=1:g2.nn
        ig = g2.t[k, i]
        k_sfc = get_k_sfc(k, nσ)
        ig_sfc = get_i_sfc(ig, nσ)
        i_sfc = mod1(i, 3)
        j = k - (k_sfc - 1)*(nσ - 1)
        Huz = H[ig_sfc]*Huσ[k] + σ[j]*Hx[k_sfc, i_sfc]*Hux[k] + σ[j]*Hy[k_sfc, i_sfc]*Huy[k]
        uzb[k, i] = Huz*b.values[g2.t[k, i]]
    end

    # element mass matrix
    M = mass_matrix(g2.el)

    # integrate
    return sum(M*uzb'*g2.J.dets)
end

"""
ε² ∫ ν [ ∂z(ux)^2 + ∂z(uy)^2] dx dy dz = ε² ∫ ν/H [ ωx^2 + ωy^2 ] dx dy dσ
"""
function KE_dissipation(m::ModelSetup3D, s::ModelState3D)
    # unpack
    ε² = m.ε²
    g2 = m.g2
    nσ = m.nσ
    ν = m.ν
    H = m.H
    ωx = s.ωx
    ωy = s.ωy

    # integrand
    function f(ξ, k)
        k_sfc = get_k_sfc(k, nσ)
        return ε²*ν(ξ, k)/H(ξ[1:2], k_sfc)*(ωx(ξ, k)^2 + ωy(ξ, k)^2)*g2.J.dets[k]
    end

    # stamp
    return sum(ref_el_quad(ξ -> f(ξ, k), g2.el) for k=1:g2.nt)
end