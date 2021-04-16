# parameters

# canonical or transport-constrained case?
#= canonical = true =#
canonical = false

τ_A = 1e2 # nondim arrest time
τ_S = 5e3 # nondim spindown time
Ek = 1/τ_S^2
S = 1/τ_A
H = τ_S # z ∈ [0, H0] ⟹ z̃ ∈ [0, H0/δ = 1/sqrt(Ek) = τ_S]
ṽ_0 = -1

# timestep
Δt̃ = minimum([τ_S/100, τ_A/100])
tSave = τ_A
α = 0.5

# number of grid points
nz̃ = 2^11 # good for anything at or below τ_S = 1e4

# grid (chebyshev, z̃ = 0 is bottom)
z̃ = @. H*(1 - cos(pi*(0:nz̃-1)/(nz̃-1)))/2

#= # bottom enhanced: =#
#= ν0 = 1e-1 =#
#= ν1 = 1 - 1e-1 =#
#= κ0 = 1e-1 =#
#= κ1 = 1 - 1e-1 =#
#= h = 10 =#
# not bottom enhanced:
ν0 = 1
ν1 = 0
κ0 = 0
κ1 = 0
h = 10
ν = @. ν0 + ν1*exp(-z̃/h)
κ = @. κ0 + κ1*exp(-z̃/h)

"""
    log(ofile, text)

Write `text` to `ofile` and print it.
"""
function log(ofile::IOStream, text::String)
    write(ofile, string(text, "\n"))
    println(text)
end

# log properties
ofile = open("out.txt", "w")
log(ofile, "\nSpin Down with Parameters\n")

log(ofile, @sprintf("nz̃ = %1.5e", nz̃))
log(ofile, @sprintf("τ_A = %1.5e", τ_A))
log(ofile, @sprintf("τ_S = %1.5e", τ_S))
log(ofile, @sprintf("H  = %1.5e", H))
log(ofile, @sprintf("S  = %1.5e", S))
log(ofile, @sprintf("κ0 = %1.5e", κ0))
log(ofile, @sprintf("κ1 = %1.5e", κ1))
log(ofile, @sprintf("ν0 = %1.5e", ν0))
log(ofile, @sprintf("ν1 = %1.5e", ν1))
log(ofile, @sprintf("h  = %1.5e", h))
log(ofile, @sprintf("v0 = %1.5e", ṽ_0))
log(ofile, @sprintf("Δt = %1.5e", Δt̃))
log(ofile, @sprintf("α  = %1.5e", α))

log(ofile, string("\nCanonical: ", canonical))

log(ofile, @sprintf("\nτ_A/τ_S  = %1.5e", τ_A/τ_S))

close(ofile)
