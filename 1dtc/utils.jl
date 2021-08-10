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
    saveCheckpoint1DTC(b, û, v̂, Px, t, i)

Save .h5 checkpoint file for state `b` at time `t`.
"""
function saveCheckpoint1DTC(b, û, v̂, Px, t, i)
    tDays = t/secsInDay
    savefile = @sprintf("checkpoint%d.h5", i)
    file = h5open(savefile, "w")
    write(file, "b", b)
    write(file, "û", û)
    write(file, "v̂", v̂)
    write(file, "U₀", U₀)
    write(file, "t", t)
    write(file, "H", H)
    write(file, "Pr", Pr)
    write(file, "f", f)
    write(file, "N", N)
    write(file, "transportConstraint", transportConstraint)
    write(file, "κ", κ)
    write(file, "ẑ", ẑ)
    write(file, "α", α)
    write(file, "θ", θ)
    close(file)
    println(savefile)
end

"""
    checkpoint = loadCheckpoint1DTC(filename)

Load .h5 checkpoint file given by `filename`.
"""
function loadCheckpoint1DTC(filename)
    file = h5open(filename, "r")
    b = read(file, "b")
    û = read(file, "û")
    v̂ = read(file, "v̂")
    U₀ = read(file, "U₀")
    t = read(file, "t")
    H = read(file, "H")
    Pr = read(file, "Pr")
    f = read(file, "f")
    N = read(file, "N")
    transportConstraint = read(file, "transportConstraint")
    κ = read(file, "κ")
    ẑ = read(file, "ẑ")
    α = read(file, "α")
    θ = read(file, "θ")
    close(file)
    return (b=b, 
            û=û, 
            v̂=v̂, 
            U₀=U₀, 
            t=t, 
            H=H, 
            Pr=Pr, 
            f=f, 
            N=N, 
            transportConstraint=transportConstraint, 
            κ=κ,
            ẑ=ẑ,
            α=α,
            θ=θ)
end
