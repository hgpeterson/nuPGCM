################################################################################
# General utility functions
################################################################################

"""
    Dξ = getDξ(ξ, L, periodic)

Compute the ξ derivative matrix.
"""
function getDξ(ξ::Array{Float64,1}, L::Float64, periodic::Bool)
    nξ = size(ξ, 1)
    
    Dξ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for i=1:nξ
        if i == 1
            # left 
            if periodic
                fd_ξ = mkfdstencil([ξ[nξ] - L, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (i, nξ, fd_ξ[1]))
                push!(Dξ, (i, 1,  fd_ξ[2]))
                push!(Dξ, (i, 2,  fd_ξ[3]))
            else
                # ghost point at ξ = 0 where derivative is zero:
                #   fd_ξ0[1]*f[0] = -fd_ξ0[2]*f[1] - fd_ξ0[3]*f[2]
                #   -> dξ(f) at ξ[1] = fd_ξ[1]*f[0] + fd_ξ[2]*f[1] + fd_ξ[3]*f[2]
                #                    = -fd_ξ[1]*(fd_ξ0[2]*f[1] + fd_ξ0[3]*f[2])/fd_ξ0[1] + fd_ξ[2]*f[1] + fd_ξ[3]*f[2]
                #                    = (fd_ξ[2] - fd_ξ[1]*fd_ξ0[2]/fd_ξ0[1]) * f[1] + (fd_ξ[3] - fd_ξ[1]fd_ξ0[3]/fd_ξ0[1]) * f[2]
                fd_ξ0 = mkfdstencil([0, ξ[1], ξ[2]], 0.0, 1) 
                fd_ξ = mkfdstencil([0, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (i, 1, fd_ξ[2] - fd_ξ[1]*fd_ξ0[2]/fd_ξ0[1]))
                push!(Dξ, (i, 2, fd_ξ[3] - fd_ξ[1]*fd_ξ0[3]/fd_ξ0[1]))

                # fd_ξ = mkfdstencil(ξ[1:3], ξ[1], 1) 
                # push!(Dξ, (i, 1, fd_ξ[1]))
                # push!(Dξ, (i, 2, fd_ξ[2]))
                # push!(Dξ, (i, 3, fd_ξ[3]))
            end
        elseif i == nξ
            # right
            if periodic
                fd_ξ = mkfdstencil([ξ[nξ-1], ξ[nξ], ξ[1] + L], ξ[nξ], 1)
                push!(Dξ, (i, nξ-1, fd_ξ[1]))
                push!(Dξ, (i, nξ,   fd_ξ[2]))
                push!(Dξ, (i, 1,    fd_ξ[3]))
            else
                fd_ξ = mkfdstencil(ξ[nξ-2:nξ], ξ[nξ], 1)
                push!(Dξ, (i, nξ-2, fd_ξ[1]))
                push!(Dξ, (i, nξ-1, fd_ξ[2]))
                push!(Dξ, (i, nξ,   fd_ξ[3]))
            end
        else
            # interior
            fd_ξ = mkfdstencil(ξ[i-1:i+1], ξ[i], 1)
            push!(Dξ, (i, i-1, fd_ξ[1]))
            push!(Dξ, (i, i,   fd_ξ[2]))
            push!(Dξ, (i, i+1, fd_ξ[3]))
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dξ = sparse((x->x[1]).(Dξ), (x->x[2]).(Dξ), (x->x[3]).(Dξ), nξ, nξ)

    return Dξ
end

"""
    Dσ = getDσ(σ)

Compute the σ derivative matrix.
"""
function getDσ(σ::Array{Float64,1})
    nσ = size(σ, 1)

    Dσ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for j=1:nσ
        if j == 1 
            # bottom 
            fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
            push!(Dσ, (j, 1, fd_σ[1]))
            push!(Dσ, (j, 2, fd_σ[2]))
            push!(Dσ, (j, 3, fd_σ[3]))
        elseif j == nσ
            # top 
            fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
            push!(Dσ, (j, nσ-2, fd_σ[1]))
            push!(Dσ, (j, nσ-1, fd_σ[2]))
            push!(Dσ, (j, nσ,   fd_σ[3]))
        else
            # interior
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            push!(Dσ, (j, j-1, fd_σ[1]))
            push!(Dσ, (j, j,   fd_σ[2]))
            push!(Dσ, (j, j+1, fd_σ[3]))
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dσ = sparse((x->x[1]).(Dσ), (x->x[2]).(Dσ), (x->x[3]).(Dσ), nσ, nσ)

    return Dσ
end

"""
    fξ = ξDerivative(m, field)

Compute dξ(`field`) in terrian-following coordinates.
"""
function ξDerivative(m::ModelSetup2DPG, field::Array{Float64})
    return m.Dξ*field
end

"""
    fσ = σDerivative(m, field)

Compute dσ(`field`) in terrian-following coordinates.
"""
function σDerivative(m::ModelSetup2DPG, field::Array{Float64,2})
    return (m.Dσ*field')'
end

"""
    fx = xDerivative(m, field)

Compute dx(`field`) in terrian-following coordinates.
Note: dx() = dξ() - dx(H)*σ*dσ()/H
"""
function xDerivative(m::ModelSetup2DPG, field::Array{Float64,2})
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
function zDerivative(m::ModelSetup2DPG, field::Array{Float64,2})
    # dσ(field)/H
    fz = σDerivative(m, field)./repeat(m.H, 1, m.nσ)
    return fz
end

"""
    u, v, w = transformFromTF(m, s)

Transform from terrain-following coordinates to cartesian coordinates.
"""
function transformFromTF(m::ModelSetup2DPG, s::ModelState2DPG)
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
function saveSetup2DPG(m::ModelSetup2DPG)
    savefile = string(outFolder, "setup.h5")
    file = h5open(savefile, "w")
    write(file, "f", m.f)
    write(file, "ξVariation", m.ξVariation)
    write(file, "L", m.L)
    write(file, "nξ", m.nξ)
    write(file, "nσ", m.nσ)
    write(file, "coords", m.coords)
    write(file, "periodic", m.periodic)
    write(file, "ξ", m.ξ)
    write(file, "σ", m.σ)
    write(file, "x", m.x)
    write(file, "z", m.z)
    write(file, "H", m.H)
    write(file, "Hx", m.Hx)
    write(file, "ν", m.ν)
    write(file, "κ", m.κ)
    write(file, "N2", m.N2)
    write(file, "Δt", m.Δt)
    close(file)
    println(savefile)

    # log 
    ofile = open(string(outFolder, "out.txt"), "w")
    logParams(ofile, "\n2D νPGCM with Parameters\n")
    logParams(ofile, @sprintf("nξ    = %d", m.nξ))
    logParams(ofile, @sprintf("nσ    = %d\n", m.nσ))
    logParams(ofile, @sprintf("L     = %d km", m.L/1000))
    logParams(ofile, @sprintf("f     = %1.1e s-1", m.f))
    logParams(ofile, @sprintf("Nₘₐₓ  = %1.1e s-1", sqrt(maximum(m.N2))))
    logParams(ofile, @sprintf("Δt    = %.2f days", m.Δt/secsInDay))
    logParams(ofile, string("\nVariations in ξ: ", m.ξVariation))
    logParams(ofile, string("Coordinates:     ", m.coords))
    logParams(ofile, string("Periodic:        ", m.periodic))
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
    ξVariation = read(file, "ξVariation")
    L = read(file, "L")
    nξ = read(file, "nξ")
    nσ = read(file, "nσ")
    coords = read(file, "coords")
    periodic = read(file, "periodic")
    ξ = read(file, "ξ")
    σ = read(file, "σ")
    x = read(file, "x")
    z = read(file, "z")
    H = read(file, "H")
    Hx = read(file, "Hx")
    ν = read(file, "ν")
    κ = read(file, "κ")
    N2 = read(file, "N2")
    Δt = read(file, "Δt")
    return ModelSetup2DPG(f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt)
end

"""
    saveState2DPG(s, iSave)

Save .h5 state file.
"""
function saveState2DPG(s::ModelState2DPG, iSave::Int64)
    savefile = @sprintf("%sstate%d.h5", outFolder, iSave)
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
    s = loadState2DPG(filename)

Load .h5 state file given by `filename`.
"""
function loadState2DPG(filename::String)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    uξ = read(file, "uξ")
    uη = read(file, "uη")
    uσ = read(file, "uσ")
    i = read(file, "i")
    close(file)
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)
    return s
end