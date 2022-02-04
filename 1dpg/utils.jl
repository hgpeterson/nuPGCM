################################################################################
# Utility functions for transport-constrained 1D PG model
################################################################################

"""
    logParams(ofile, text)

Write `text` to `ofile` and print it.
"""
function logParams(ofile::IOStream, text::String)
    write(ofile, string(text, "\n"))
    println(text)
end

"""
    saveSetup1DPG(m)

Save .h5 file for parameters.
"""
function saveSetup1DPG(m::ModelSetup1DPG)
    savefile = string(outFolder, "setup.h5")
    file = h5open(savefile, "w")
    write(file, "bl", m.bl)
    write(file, "f", m.f)
    write(file, "nz", m.nz)
    write(file, "z", m.z)
    write(file, "H", m.H)
    write(file, "θ", m.θ)
    write(file, "ν", m.ν)
    write(file, "κ", m.κ)
    write(file, "κ_z", m.κ_z)
    write(file, "N2", m.N2)
    write(file, "Δt", m.Δt)
    write(file, "transportConstraint", m.transportConstraint)
    write(file, "U", m.U)
    close(file)
    println(savefile)

    # log 
    ofile = open(string(outFolder, "out.txt"), "w")
    if m.bl
        logParams(ofile, "\n1D BL PG Model with Parameters\n")
    else
        logParams(ofile, "\n1D PG Model with Parameters\n")
    end
    logParams(ofile, @sprintf("nz    = %d", m.nz))
    logParams(ofile, @sprintf("H     = %d km", m.H/1000))
    logParams(ofile, @sprintf("θ     = %1.1e rad", m.θ))
    logParams(ofile, @sprintf("f     = %1.1e s-1", m.f))
    logParams(ofile, @sprintf("N     = %1.1e s-1", sqrt(m.N2)))
    logParams(ofile, @sprintf("S     = %1.1e", m.N2/m.f^2*tan(m.θ)^2))
    logParams(ofile, @sprintf("Δt    = %.2f days", m.Δt/secsInDay))
    logParams(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1]/abs(m.f))))
    logParams(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.z[2] - m.z[1]))
    close(ofile)
end

"""
    m = loadSetup1DPG(filename)

Load .h5 setup file given by `filename`.
"""
function loadSetup1DPG(filename::String)
    file = h5open(filename, "r")
    bl = read(file, "bl")
    f = read(file, "f")
    nz = read(file, "nz")
    z = read(file, "z")
    H = read(file, "H")
    θ = read(file, "θ")
    ν = read(file, "ν")
    κ = read(file, "κ")
    κ_z = read(file, "κ_z")
    N2 = read(file, "N2")
    Δt = read(file, "Δt")
    transportConstraint = read(file, "transportConstraint")
    U = read(file, "U")
    close(file)
    return ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U)
end

"""
    saveState1DPG(s, iSave)

Save .h5 checkpoint file for state.
"""
function saveState1DPG(s::ModelState1DPG, iSave::Int64)
    savefile = @sprintf("%sstate%d.h5", outFolder, iSave)
    file = h5open(savefile, "w")
    write(file, "b", s.b)
    write(file, "χ", s.χ)
    write(file, "u", s.u)
    write(file, "v", s.v)
    write(file, "i", s.i)
    close(file)
    println(savefile)
end

"""
    s = loadState1DPG(filename)

Load .h5 state file given by `filename`.
"""
function loadState1DPG(filename::String)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    u = read(file, "u")
    v = read(file, "v")
    i = read(file, "i")
    close(file)
    return ModelState1DPG(b, χ, u, v, i)
end

"""
    χB = boundaryCorrection(m, χI, z)

Compute BL correction to `χI` on grid `z`.
"""
function boundaryCorrection(m::ModelSetup1DPG, χI::Vector{Float64}, z::Vector{Float64})
    δ, μ, S, q = get_BL_params(m)
    c1 = -m.U[1]/(1 + μ*S) - χI[1]
    dχIdz = differentiate_pointwise(χI[1:3], z[1:3], z[1], 1)
    c2 = c1 - dχIdz/q
    return @. m.U[1]/(1 + μ*S) + exp(-q*z)*(c1*cos(q*z) + c2*sin(q*z))
end

"""
    δ, μ, S, q = get_BL_params(m)

Compute classical flat-bottom Ekman layer thickness `δ`,
Prandtl number `μ`, slope Burger number `S`, and BL thickness `q`.
"""
function get_BL_params(m::ModelSetup1DPG)
    δ = sqrt(2*m.ν[1]/abs(m.f))
    μ = m.ν[1]/m.κ[1]
    S = m.N2/m.f^2*tan(m.θ)^2
    q = 1/δ * (1 + μ*S)^(1/4)
    return δ, μ, S, q
end

"""
    χ, b = constructFullSolution(m, s, z)

Construct full solutions `χ` = χI + χB and `b` = bI + bB from BL theory.
The full solutions exist on the new grid `z`.
"""
function constructFullSolution(m::ModelSetup1DPG, s::ModelState1DPG, z::Vector{Float64})
    # interior vars
    bI = s.b
    χI = m.U .- differentiate(bI, m.z)*tan(m.θ).*m.ν/m.f^2

    # interpolate onto new grid 
    χI_fine = Spline1D(m.z .- m.z[1], χI)(z)
    bI_fine = Spline1D(m.z .- m.z[1], bI)(z)

    # BL correction
    χB = boundaryCorrection(m, χI_fine, z)
    bB = cumtrapz(χB*m.N2*tan(m.θ)/m.κ[1], z) .- trapz(χB*m.N2*tan(m.θ)/m.κ[1], z)

    # full sol
    χ = χI_fine + χB
    b = bI_fine + bB
    return χ, b
end