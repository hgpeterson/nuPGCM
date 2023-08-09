"""
    χ, b = get_full_soln(m, s, z, ix)

Construct full solutions `χ` = χI + χB and `b` = bI + bB at ξ = ξ[ix] from BL theory.
The full solutions exist on the new grid `z`.
"""
function get_full_soln(m::ModelSetup2D, s::ModelState2D, z, ix)
    # interior vars
    bI = s.b[ix, :]
    χI = s.χ[ix, :]

    # BL thickness 
    bIξ = ∂ξ(m, s.b)
    δ = sqrt(2*m.ν[ix, 1]/abs(m.f))
    μ = m.ν[ix, 1]/m.κ[ix, 1]
    S = -1/m.f^2 * m.Hx[ix]*bIξ[ix, 1]
    q = 1/δ * (1 + μ*S)^(1/4)

    # interpolate onto new grid 
    χI_fine = Spline1D(m.z[ix, :] .- m.z[ix, 1], χI)(z)
    bI_fine = Spline1D(m.z[ix, :] .- m.z[ix, 1], bI)(z)

    # BL correction
    χB = @. -χI[1]*exp(-q*z)*(cos(q*z) + sin(q*z))
    bB = cumtrapz(χB*bIξ[ix, 1]/m.κ[ix, 1], z) .- trapz(χB*bIξ[ix, 1]/m.κ[ix, 1], z) 

    # full sol
    χ = χI_fine + χB
    b = bI_fine + bB
    return χ, b
end