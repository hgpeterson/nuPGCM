# parameters

# canonical or transport-constrained case?
canonical = true
# canonical = false

τ_A = 2e0 # nondim arrest time
τ_S = 1e2 # nondim spindown time
Ek = 1/τ_S^2
S = 1/τ_A
H = τ_S # z ∈ [0, H0] ⟹ z̃ ∈ [0, H0/δ = 1/sqrt(Ek) = τ_S]
ṽ_0 = -1
N = 1 

# timestep
Δt̃ = minimum([τ_S/100, τ_A/100])
tSave = τ_A
α = 0.5

# number of grid points
nz̃ = 2^11 # good for anything at or below τ_S = 1e4

# grid (chebyshev, z̃ = 0 is bottom)
z̃ = @. H*(1 - cos(pi*(0:nz̃-1)/(nz̃-1)))/2

# bottom enhanced:
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
h = 1
ν = @. ν0 + ν1*exp(-z̃/h)
κ = @. κ0 + κ1*exp(-z̃/h)

"""
    logOut(ofile, text)

Write `text` to `ofile` and print it.
"""
function logOut(ofile::IOStream, text::String)
    write(ofile, string(text, "\n"))
    println(text)
end

# log properties
ofile = open("out.txt", "w")
logOut(ofile, "\nSpin Down with Parameters\n")

logOut(ofile, @sprintf("nz̃ = %1.5e", nz̃))
logOut(ofile, @sprintf("τ_A = %1.5e", τ_A))
logOut(ofile, @sprintf("τ_S = %1.5e", τ_S))
logOut(ofile, @sprintf("H  = %1.5e", H))
logOut(ofile, @sprintf("S  = %1.5e", S))
logOut(ofile, @sprintf("κ0 = %1.5e", κ0))
logOut(ofile, @sprintf("κ1 = %1.5e", κ1))
logOut(ofile, @sprintf("ν0 = %1.5e", ν0))
logOut(ofile, @sprintf("ν1 = %1.5e", ν1))
logOut(ofile, @sprintf("ṽ_0 = %1.5e", ṽ_0))
logOut(ofile, @sprintf("N = %1.5e", N))
logOut(ofile, @sprintf("h  = %1.5e", h))
logOut(ofile, @sprintf("Δt = %1.5e", Δt̃))
logOut(ofile, @sprintf("α  = %1.5e", α))

logOut(ofile, string("\nCanonical: ", canonical))

logOut(ofile, @sprintf("\nτ_A/τ_S  = %1.5e", τ_A/τ_S))

close(ofile)