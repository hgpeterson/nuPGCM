# parameters (as in RC20)
L = 2e6
H0 = 2e3
f = -5.5e-5
N = 1e-3
# as in CF18
β = 2e-11
r = 0.1*β*L
#= r = 1.2e-5 =#

# turn on/off variations in ξ
ξVariation = true

# topography: sine
amp =  0.4*H0
H(x) = H0 - amp*sin(2*π*x/L - π/2)
Hx(x) = -2*π/L*amp*cos(2*π*x/L - π/2)

# number of grid points
nξ = 2^8 + 1 
nσ = 2^8

# domain in terrain-following (ξ, σ) space
dξ = dx = L/nξ
ξ = 0:dξ:(L - dξ)
σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2 # chebyshev 
ξξ = repeat(ξ, 1, nσ)
σσ = repeat(σ', nξ, 1)
dσ = zeros(nξ, nσ)
dσ[:, 1:end-1] = σσ[:, 2:end] - σσ[:, 1:end-1]
dσ[:, end] = dσ[:, end-1]

# domain in physical (x, z) space (2D arrays)
x = repeat(ξ, 1, nσ)
z = repeat(σ', nξ, 1).*repeat(H.(ξ), 1, nσ)

# arrays of sin(θ) and cos(θ) for 1D solutions
sinθ = @. -Hx(ξξ)/sqrt(1 + Hx(ξξ)^2)
cosθ = @. 1/sqrt(1 + Hx(ξξ)^2) 
θ = asin.(sinθ[:, 1])

# diffusivity
κ0 = 6e-7
κ1 = 2e-5
h = 200
# not bottom enhanced:
# κ0 = 6e-7
# κ1 = 0
# h = 200
κ = @. κ0 + κ1*exp(-(z + H(x))/h)
    
# timestepping
secsInDay = 86400
secsInYear = 360*86400
Δt = secsInDay
tPlot = 3*secsInYear
tSave = 3*secsInYear

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
log(ofile, "\nRayleigh Drag PGSolver with Parameters\n")

log(ofile, @sprintf("nξ = %d", nξ))
log(ofile, @sprintf("nσ = %d\n", nσ))
log(ofile, @sprintf("L  = %d km", L/1000))
log(ofile, @sprintf("H0 = %d m", H0))
log(ofile, @sprintf("f  = %1.1e s-1", f))
log(ofile, @sprintf("N  = %1.1e s-1", N))
log(ofile, @sprintf("r  = %1.1e s-1", r))
log(ofile, @sprintf("κ0 = %1.1e m2 s-1", κ0))
log(ofile, @sprintf("κ1 = %1.1e m2 s-1", κ1))
log(ofile, @sprintf("h  = %d m", h))
log(ofile, @sprintf("Δt = %1.1e s", Δt))

log(ofile, string("\nVariations in ξ: ", ξVariation))

iξ = argmin(abs.(ξ .- L/4))
q1 = sqrt(r*N^2*sinθ[iξ, 1]^2/(cosθ[iξ, 1]^2*κ[iξ, 1]*(f^2 + r^2)))
log(ofile, @sprintf("\nBL thickness ~ %1.2f m", q1^-1))
log(ofile, @sprintf(" z[2] - z[1] ~ %1.2f m\n", H0*(σ[2] - σ[1])))
close(ofile)