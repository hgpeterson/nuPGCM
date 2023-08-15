"""
    χB = get_BL_correction(m, χI, z)

Compute BL correction to `χI` on grid `z`.
"""
function get_BL_correction(m::ModelSetup1D, χI, z)
    δ, μ, S, q = get_BL_params(m)
    return @. -χI[1]*exp(-q*z)*(cos(q*z) + sin(q*z))
end

"""
    δ, μ, S, q = get_BL_params(m)

Compute classical flat-bottom Ekman layer thickness `δ`,
Prandtl number `μ`, slope Burger number `S`, and BL thickness `q`.
"""
function get_BL_params(m::ModelSetup1D)
    δ = sqrt(2*m.ν[1]/abs(m.f))
    μ = m.ν[1]/m.κ[1]
    S = m.N2/m.f^2*tan(m.θ)^2
    q = 1/δ * (1 + μ*S)^(1/4)
    return δ, μ, S, q
end

"""
    χ, b = get_full_soln(m, s, z)

Construct full solutions `χ` = χI + χB and `b` = bI + bB from BL theory.
The full solutions exist on the new grid `z`.
"""
function get_full_soln(m::ModelSetup1D, s::ModelState1D, z)
    # interior vars
    bI = s.b
    χI = m.U .- differentiate(bI, m.z)*tan(m.θ).*m.ν/m.f^2

    # interpolate onto new grid 
    χI_fine = Spline1D(m.z .- m.z[1], χI)(z)
    bI_fine = Spline1D(m.z .- m.z[1], bI)(z)

    # BL correction
    χB = get_BL_correction(m, χI_fine, z)
    bB = cumtrapz(χB*m.N2*tan(m.θ)/m.κ[1], z) .- trapz(χB*m.N2*tan(m.θ)/m.κ[1], z)

    # full sol
    χ = χI_fine + χB
    b = bI_fine + bB
    return χ, b
end