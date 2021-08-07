################################################################################
# General utility functions
################################################################################

using HDF5
include("setup.jl")

"""
    fξ = ξDerivative(m, field)

Compute dξ(`field`) in terrian-following coordinates.
"""
function ξDerivative(m::ModelSetup, field::Array{Float64,1})
    # allocate
    fξ = zeros(m.nξ, 1)

    # uniform grid spacing
    dξ = m.ξ[2] - m.ξ[1]

    # dξ(field): use the fact that ξ is evenly spaced and periodic
    fξ[2:end-1] = (field[3:end] - field[1:end-2])/(2*dξ)
    fξ[1] = (field[2] - field[end])/(2*dξ)
    fξ[end] = (field[1] - field[end-1])/(2*dξ)

    return fξ
end
function ξDerivative(m::ModelSetup, field::Array{Float64,2})
    return reshape(m.Dξ*field[:], m.nξ, m.nσ)
end

"""
    fσ = σDerivative(m, field)

Compute dσ(`field`) in terrian-following coordinates.
"""
function σDerivative(m::ModelSetup, field::Array{Float64,2})
    return reshape(m.Dσ*field[:], m.nξ, m.nσ)
end

"""
    fx = xDerivative(m, field)

Compute dx(`field`) in terrian-following coordinates.
Note: dx() = dξ() - dx(H)*σ*dσ()/H
"""
function xDerivative(m::ModelSetup, field::Array{Float64,2})
    # dξ(field)
    fx = ξDerivative(m, field)

    # -dx(H)*σ*dσ(field)/H
    fx -= repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*σDerivative(m, field)

    return fx
end

"""
    fz = zDerivative(m, field)

Compute dz(`field`) in terrian-following coordinates.
Note: dz() = dσ()/H
"""
function zDerivative(m::ModelSetup, field::Array{Float64,2})
    # dσ(field)/H
    fz = σDerivative(m, field)./repeat(m.H, 1, m.nσ)
    return fz
end

"""
    u, v, w = transformFromTF(m, s)

Transform from terrain-following coordinates to cartesian coordinates.
"""
function transformFromTF(m::ModelSetup, s::ModelState)
    u = s.uξ
    v = s.uη
    w = s.uσ.*repeat(m.H, 1, m.nσ) + repeat(m.σ', m.nξ, 1).*repeat(m.Hx, 1, m.nσ).*s.uξ
    return u, v, w
end

"""
    logParams(ofile, text)

Write `text` to `ofile` and print it.
"""
function logParams(ofile::IOStream, text::String)
    write(ofile, string(text, "\n"))
    println(text)
end

"""
    saveSetup2DPG(m)

Save .h5 file for parameters.
"""
function saveSetup2DPG(m::ModelSetup)
    savefile = "setup.h5"
    file = h5open(savefile, "w")
    write(file, "f", m.f)
    write(file, "N", m.N)
    write(file, "ξVariation", m.ξVariation)
    write(file, "L", m.L)
    write(file, "nξ", m.nξ)
    write(file, "nσ", m.nσ)
    write(file, "ξ", m.ξ)
    write(file, "σ", m.σ)
    write(file, "x", m.x)
    write(file, "z", m.z)
    write(file, "H", m.H)
    write(file, "Hx", m.Hx)
    write(file, "ν", m.ν)
    write(file, "κ", m.κ)
    write(file, "Δt", m.Δt)
    close(file)
    println(savefile)

    # log 
    ofile = open("out.txt", "w")
    logParams(ofile, "\n2D νPGCM with Parameters\n")
    logParams(ofile, @sprintf("nξ = %d", m.nξ))
    logParams(ofile, @sprintf("nσ = %d\n", m.nσ))
    logParams(ofile, @sprintf("L  = %d km", m.L/1000))
    logParams(ofile, @sprintf("f  = %1.1e s-1", m.f))
    logParams(ofile, @sprintf("N  = %1.1e s-1", m.N))
    logParams(ofile, @sprintf("Δt = %.2f days", m.Δt/secsInDay))
    logParams(ofile, string("\nVariations in ξ: ", m.ξVariation))
    logParams(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1, 1]/abs(m.f))))
    logParams(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.z[1, 2] - m.z[1, 1]))
    close(ofile)
end

"""
    m = loadSetup2DPG(filename)

Load .h5 setup file given by `filename`.
"""
function loadSetup2DPG(filename::String)
    file = h5open(filename, "r")
    f = read(file, "f")
    N = read(file, "N")
    ξVariation = read(file, "ξVariation")
    L = read(file, "L")
    nξ = read(file, "nξ")
    nσ = read(file, "nσ")
    ξ = read(file, "ξ")
    σ = read(file, "σ")
    x = read(file, "x")
    z = read(file, "z")
    H = read(file, "H")
    Hx = read(file, "Hx")
    ν = read(file, "ν")
    κ = read(file, "κ")
    Δt = read(file, "Δt")
    return ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, x, z, H, Hx, ν, κ, Δt)
end

"""
    saveCheckpoint2DPG(s, iSave)

Save .h5 checkpoint file for state.
"""
function saveCheckpoint2DPG(s::ModelState, iSave::Int64)
    savefile = @sprintf("checkpoint%d.h5", iSave)
    file = h5open(savefile, "w")
    write(file, "b", s.b)
    write(file, "χ", s.χ)
    write(file, "uξ", s.uξ)
    write(file, "uη", s.uη)
    write(file, "uσ", s.uσ)
    write(file, "i", s.i)
    close(file)
    println(savefile)
end

"""
    s = loadCheckpoint2DPG(filename)

Load .h5 checkpoint file given by `filename`.
"""
function loadCheckpoint2DPG(filename::String)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    uξ = read(file, "uξ")
    uη = read(file, "uη")
    uσ = read(file, "uσ")
    i = read(file, "i")
    close(file)
    s = ModelState(b, χ, uξ, uη, uσ, i)
    return s
end
