################################################################################
# Utility functions for nondimensional spindown
################################################################################

"""
    saveCheckpointSpinDown(ũ, ṽ, b̃, Px, t̃, i)

Save .h5 checkpoint file for state.
"""
function saveCheckpointSpinDown(ũ, ṽ, b̃, Px, t̃, i)
    savefile = @sprintf("checkpoint%d.h5", i)
    file = h5open(savefile, "w")
    write(file, "ũ", ũ)
    write(file, "ṽ", ṽ)
    write(file, "b̃", b̃)
    write(file, "Px", Px)
    write(file, "t̃", t̃)
    write(file, "H", H)
    write(file, "Pr", Pr)
    write(file, "S", S)
    write(file, "canonical", canonical)
    write(file, "bottomIntense", bottomIntense)
    write(file, "κ", κ)
    write(file, "κ0", κ0)
    write(file, "κ1", κ1)
    write(file, "h", h)
    write(file, "α", α)
    write(file, "z̃", z̃)
    close(file)
    println(savefile)
end

"""
    checkpoint = = loadCheckpointSpinDown(filename)

Load .h5 checkpoint file given by `filename`.
"""
function loadCheckpointSpinDown(filename)
    file = h5open(filename, "r")
    ũ = read(file, "ũ")
    ṽ = read(file, "ṽ")
    b̃ = read(file, "b̃")
    Px = read(file, "Px")
    t̃ = read(file, "t̃")
    H = read(file, "H")
    Pr = read(file, "Pr")
    S = read(file, "S")
    canonical = read(file, "canonical")
    bottomIntense = read(file, "bottomIntense")
    κ = read(file, "κ")
    κ0 = read(file, "κ0")
    κ1 = read(file, "κ1")
    h = read(file, "h")
    α = read(file, "α")
    z̃ = read(file, "z̃")
    close(file)
    return (ũ=ũ, 
            ṽ=ṽ, 
            b̃=b̃, 
            Px=Px, 
            t̃=t̃, 
            H=H, 
            Pr=Pr, 
            S=S, 
            canonical=canonical, 
            bottomIntense=bottomIntense, 
            κ=κ, 
            κ0=κ0, 
            κ1=κ1, 
            h=h, 
            α=α,
            z̃=z̃)
end
