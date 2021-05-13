################################################################################
# Utility functions for transport-constrained 1D PG model
################################################################################

"""
    u, w = rotate(û)

Rotate `û` into physical coordinate components `u` and `w`.
"""
function rotate(û)
    u = û*cos(θ)
    w = û*sin(θ)
    return u, w
end

"""
    saveCheckpointRot(b, χ, û, v̂, U, t, i)

Save .h5 checkpoint file for state `b` at time `t`.
"""
function saveCheckpointRot(b, χ, û, v̂, U, t, i)
    tDays = t/secsInDay
    savefile = @sprintf("checkpoint%d.h5", i)
    file = h5open(savefile, "w")
    write(file, "b", b)
    write(file, "χ", χ)
    write(file, "û", û)
    write(file, "v̂", v̂)
    write(file, "U", U)
    write(file, "U₀", U₀)
    write(file, "t", t)
    write(file, "H", H)
    write(file, "Pr", Pr)
    write(file, "f", f)
    write(file, "N", N)
    write(file, "transportConstraint", transportConstraint)
    write(file, "κ", κ)
    write(file, "ẑ", ẑ)
    write(file, "θ", θ)
    close(file)
    println(savefile)
end

"""
    checkpoint = loadCheckpointRot(filename)

Load .h5 checkpoint file given by `filename`.
"""
function loadCheckpointRot(filename)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    û = read(file, "û")
    v̂ = read(file, "v̂")
    U = read(file, "U")
    U₀ = read(file, "U₀")
    t = read(file, "t")
    H = read(file, "H")
    Pr = read(file, "Pr")
    f = read(file, "f")
    N = read(file, "N")
    transportConstraint = read(file, "transportConstraint")
    κ = read(file, "κ")
    ẑ = read(file, "ẑ")
    θ = read(file, "θ")
    close(file)
    return (b=b, 
            χ=χ, 
            û=û, 
            v̂=v̂, 
            U=U, 
            U₀=U₀, 
            t=t, 
            H=H, 
            Pr=Pr, 
            f=f, 
            N=N, 
            transportConstraint=transportConstraint, 
            κ=κ,
            ẑ=ẑ,
            θ=θ)
end
