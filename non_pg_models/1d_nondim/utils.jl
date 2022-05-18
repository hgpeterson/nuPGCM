################################################################################
# Utility functions for nondimensional transport-constrained 1D model
################################################################################

using Printf

"""
    saveCheckpoint1DTCNondim(ũ, ṽ, b̃, P̃x̃, t̃, i)

Save .h5 checkpoint file for state.
"""
function saveCheckpoint1DTCNondim(ũ, ṽ, b̃, P̃x̃, t̃, i)
    savefile = @sprintf("checkpoint%d.h5", i)
    file = h5open(savefile, "w")
    write(file, "ũ", ũ)
    write(file, "ṽ", ṽ)
    write(file, "b̃", b̃)
    write(file, "P̃x̃", P̃x̃)
    write(file, "t̃", t̃)
    write(file, "H", H)
    write(file, "S", S)
    write(file, "canonical", canonical)
    write(file, "ν", ν)
    write(file, "ν0", ν0)
    write(file, "ν1", ν1)
    write(file, "κ", κ)
    write(file, "κ0", κ0)
    write(file, "κ1", κ1)
    write(file, "h", h)
    write(file, "N", N)
    write(file, "α", α)
    write(file, "z̃", z̃)
    close(file)
    println(savefile)
end

"""
    checkpoint = = loadCheckpoint1DTCNondim(filename)

Load .h5 checkpoint file given by `filename`.
"""
function loadCheckpoint1DTCNondim(filename)
    file = h5open(filename, "r")
    ũ = read(file, "ũ")
    ṽ = read(file, "ṽ")
    b̃ = read(file, "b̃")
    P̃x̃ = read(file, "P̃x̃")
    t̃ = read(file, "t̃")
    H = read(file, "H")
    S = read(file, "S")
    canonical = read(file, "canonical")
    ν = read(file, "ν")
    ν0 = read(file, "ν0")
    ν1 = read(file, "ν1")
    κ = read(file, "κ")
    κ0 = read(file, "κ0")
    κ1 = read(file, "κ1")
    h = read(file, "h")
    N = read(file, "N")
    α = read(file, "α")
    z̃ = read(file, "z̃")
    close(file)
    return (ũ=ũ, 
            ṽ=ṽ, 
            b̃=b̃, 
            P̃x̃=P̃x̃, 
            t̃=t̃, 
            H=H, 
            S=S, 
            canonical=canonical, 
            ν=ν, 
            ν0=ν0, 
            ν1=ν1, 
            κ=κ, 
            κ0=κ0, 
            κ1=κ1, 
            h=h, 
            N=N, 
            α=α,
            z̃=z̃)
end