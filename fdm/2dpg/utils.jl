################################################################################
# Utility functions for 2D νPGCM 
################################################################################

"""
    fξ = ξDerivativeTF(m, field)

Compute dξ(`field`) in terrian-following coordinates.
"""
function ξDerivativeTF(m::ModelSetup, field::Array{Float64,2})
    # allocate
    fξ = zeros(m.nξ, m.nσ)

    # uniform grid spacing
    dξ = m.ξ[2] - m.ξ[1]

    # dξ(field)
    for j=1:m.nσ
        # use the fact that ξ is evenly spaced and periodic
        fξ[2:end-1, j] = (field[3:end, j] - field[1:end-2, j])/(2*dξ)
        fξ[1, j] = (field[2, j] - field[end, j])/(2*dξ)
        fξ[end, j] = (field[1, j] - field[end-1, j])/(2*dξ)
    end

    return fξ
end
function ξDerivativeTF(m::ModelSetup, field::Array{Float64,1})
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

"""
    fσ = σDerivativeTF(m, field)

Compute dσ(`field`) in terrian-following coordinates.
"""
function σDerivativeTF(m::ModelSetup, field::Array{Float64,2})
    # allocate
    fσ = zeros(m.nξ, m.nσ)

    # dσ(field)
    for i=1:m.nξ
        fσ[i, :] = differentiate(field[i, :], m.σ)
    end

    return fσ
end

"""
    fx = xDerivativeTF(m, field)

Compute dx(`field`) in terrian-following coordinates.
Note: dx() = dξ() - dx(H)*σ*dσ()/H
"""
function xDerivativeTF(m::ModelSetup, field::Array{Float64,2})
    # dξ(field)
    fx = ξDerivativeTF(m, field)

    # -dx(H)*σ*dσ(field)/H
    fx -= repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*σDerivativeTF(m, field)

    return fx
end

"""
    fz = zDerivativeTF(m, field)

Compute dz(`field`) in terrian-following coordinates.
Note: dz() = dσ()/H
"""
function zDerivativeTF(m::ModelSetup, field::Array{Float64,2})
    # dσ(field)/H
    fz = σDerivativeTF(m, field)./repeat(m.H, 1, m.nσ)
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
    write(file, "sol_U", m.sol_U)
    close(file)
    println(savefile)
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
    inversionLHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}}(undef, nξ)
    sol_U = read(file, "sol_U")
    return ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, x, z, H, Hx, ν, κ, Δt, inversionLHSs, sol_U)
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
