################################################################################
# Utility functions for transport-constrained 1D model
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
    saveSetup1D(m)

Save .h5 file for parameters.
"""
function saveSetup1D(m::ModelSetup1D)
    savefile = string(outFolder, "setup.h5")
    file = h5open(savefile, "w")
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
    write(file, "Uamp", m.Uamp)
    write(file, "Uper", m.Uper)
    close(file)
    println(savefile)

    # log 
    ofile = open(string(outFolder, "out.txt"), "w")
    logParams(ofile, "\n1D Model with Parameters\n")
    logParams(ofile, @sprintf("nz    = %d", m.nz))
    logParams(ofile, @sprintf("H     = %d km", m.H/1000))
    logParams(ofile, @sprintf("θ     = %1.1e rad", m.θ))
    logParams(ofile, @sprintf("f     = %1.1e s-1", m.f))
    logParams(ofile, @sprintf("N     = %1.1e s-1", sqrt(m.N2)))
    logParams(ofile, @sprintf("S     = %1.1e", m.N2/m.f^2*tan(m.θ)^2))
    logParams(ofile, @sprintf("Δt    = %.2f days", m.Δt/secsInDay))
    logParams(ofile, @sprintf("Uamp  = %1.1e m2 s-1", m.Uamp))
    logParams(ofile, @sprintf("Uper  = %.2f days", m.Uper/secsInDay))
    logParams(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1]/abs(m.f))))
    logParams(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.z[2] - m.z[1]))
    close(ofile)
end

"""
    m = loadSetup1D(filename)

Load .h5 setup file given by `filename`.
"""
function loadSetup1D(filename::String)
    file = h5open(filename, "r")
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
    Uamp = read(file, "Uamp")
    Uper = read(file, "Uper")
    close(file)
    return ModelSetup1D(f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U, Uamp, Uper)
end

"""
    saveState1D(s, iSave)

Save .h5 checkpoint file for state.
"""
function saveState1D(s::ModelState1D, iSave::Int64)
    savefile = @sprintf("%sstate%d.h5", outFolder, iSave)
    file = h5open(savefile, "w")
    write(file, "b", s.b)
    write(file, "u", s.u)
    write(file, "v", s.v)
    write(file, "∂ₓP", s.∂ₓP)
    write(file, "i", s.i)
    close(file)
    println(savefile)
end

"""
    s = loadState1D(filename)

Load .h5 state file given by `filename`.
"""
function loadState1D(filename::String)
    file = h5open(filename, "r")
    b = read(file, "b")
    u = read(file, "u")
    v = read(file, "v")
    ∂ₓP = read(file, "∂ₓP")
    i = read(file, "i")
    close(file)
    return ModelState1D(b, u, v, ∂ₓP, i)
end