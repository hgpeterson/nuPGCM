################################################################################
# Useful functions for terrain following coordinates
################################################################################

"""
    fξ = ξDerivativeTF(field)

Compute dξ(`field`) in terrian-following coordinates.
"""
function ξDerivativeTF(field)
    # allocate
    fξ = zeros(nξ, nσ)

    # dξ(field)
    for j=1:nσ
        # use the fact that ξ is evenly spaced and periodic
        fξ[2:end-1, j] = (field[3:end, j] - field[1:end-2, j])/(2*dξ)
        fξ[1, j] = (field[2, j] - field[nξ, j])/(2*dξ)
        fξ[end, j] = (field[1, j] - field[end-1, j])/(2*dξ)
    end

    return fξ
end

"""
    fσ = σDerivativeTF(field)

Compute dσ(`field`) in terrian-following coordinates.
"""
function σDerivativeTF(field)
    # allocate
    fσ = zeros(nξ, nσ)

    # dσ(field)
    for i=1:nξ
        fσ[i, :] .+= differentiate(field[i, :], σ)
    end

    return fσ
end

"""
    fx = xDerivativeTF(field)

Compute dx(`field`) in terrian-following coordinates.
Note: dx() = dξ() - dx(H)*σ*dσ()/H
"""
function xDerivativeTF(field)
    # dξ(field)
    fx = ξDerivativeTF(field)

    # -dx(H)*σ*dσ(field)/H
    fx -= Hx.(x).*σσ.*σDerivativeTF(field)./H.(x)

    return fx
end

"""
    fz = zDerivativeTF(field)

Compute dz(`field`) in terrian-following coordinates.
Note: dz() = dσ()/H
"""
function zDerivativeTF(field)
    # dσ(field)/H
    fz = σDerivativeTF(field)./H.(x)
    return fz
end

"""
    u, v, w = transformFromTF(uξ, uη, uσ)

Transform from terrain-following coordinates to cartesian coordinates.
"""
function transformFromTF(uξ, uη, uσ)
    u = uξ
    v = uη
    w = uσ.*H.(x) + σσ.*Hx.(x).*u
    return u, v, w
end

"""
    uξ, uη, uσ = transformToTF(u, v, w)

Transform from cartesian coordinates to terrain-following coordinates.
"""
function transformToTF(u, v, w)
    uξ = u
    uη = v
    uσ = (w - σσ.*Hx.(x).*u)./H.(x)
    return uξ, uη, uσ
end

"""
    saveCheckpointTF(b, chi, uξ, uη, uσ, U, t)

Save .h5 checkpoint file for state `b` at time `t`.
"""
function saveCheckpointTF(b, chi, uξ, uη, uσ, U, t)
    tDays = t/86400
    savefile = @sprintf("checkpoint%d.h5", tDays)
    file = h5open(savefile, "w")
    write(file, "x", x)
    write(file, "z", z)
    write(file, "b", b)
    write(file, "chi", chi)
    write(file, "uξ", uξ)
    write(file, "uη", uη)
    write(file, "uσ", uσ)
    write(file, "U", U)
    write(file, "t", t)
    write(file, "L", L)
    write(file, "H0", H0)
    write(file, "Pr", Pr)
    write(file, "f", f)
    write(file, "N", N)
    write(file, "ξVariation", ξVariation)
    write(file, "κ", κ)
    close(file)
    println(savefile)
end

"""
    checkpoint = loadCheckpointTF(filename)

Load .h5 checkpoint file given by `filename`.
"""
function loadCheckpointTF(filename)
    file = h5open(filename, "r")
    #= x = read(file, "x") =#
    x = 1
    #= z = read(file, "z") =#
    z = 1
    b = read(file, "b")
    chi = read(file, "chi")
    uξ = read(file, "uξ")
    uη = read(file, "uη")
    uσ = read(file, "uσ")
    U = read(file, "U")
    t = read(file, "t")
    L = read(file, "L")
    H0 = read(file, "H0")
    Pr = read(file, "Pr")
    f = read(file, "f")
    N = read(file, "N")
    ξVariation, = read(file, "ξVariation")
    κ = read(file, "κ")
    close(file)
    return (x=x,
            z=z,
            b=b, 
            chi=chi, 
            uξ=uξ, 
            uη=uη, 
            uσ=uσ, 
            U=U, 
            t=t, 
            L=L, 
            H0=H0, 
            Pr=Pr, 
            f=f, 
            N=N, 
            ξVariation=ξVariation, 
            κ=κ)
end
