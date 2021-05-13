# parameters (as in RC20)
Pr = 1e0
f = -5.5e-5
N = 1e-3

# turn on/off variations in ξ
ξVariation = true

# topography: sine
L = 2e6
H0 = 2e3
amp =  0.4*H0
H(x) = H0 - amp*sin(2*π*x/L - π/2)
Hx(x) = -2*π/L*amp*cos(2*π*x/L - π/2)

# topography: skew gaussian
#= L = 2e6 =#
#= H0 = 2.5e3 =#
#= amp = 0.65*H0 =#
#= ϕ(s) = exp(-s^2/2) =#
#= Φ(s) = 1/2*(1 + erf(s/√2)) =#
#= α = 3 =#
#= μ = L/3 =#
#= ω = L/5 =#
#= H(x) = H0 - amp*ϕ((x - μ)/ω)*Φ(α*(x - μ)/ω) =#
#= Hx(x) = -amp/ω*(α/sqrt(2π)*ϕ(α*√2*(x - μ)/ω)*ϕ((x - μ)/ω) - (x - μ)/ω*ϕ((x - μ)/ω)*Φ(α*(x - μ)/ω)) =#

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

#= plot(ξ, -H.(ξ)) =# 
#= ylim([-H0, 0]) =#
#= savefig("topo.png") =#
#= println(Hx(0)) =#
#= error() =#

# domain in physical (x, z) space (2D arrays)
x = repeat(ξ, 1, nσ)
z = repeat(σ', nξ, 1).*repeat(H.(ξ), 1, nσ)

# arrays of sin(θ) and cos(θ) for 1D solutions
sinθ = @. -Hx(ξξ)/sqrt(1 + Hx(ξξ)^2)
cosθ = @. 1/sqrt(1 + Hx(ξξ)^2) 
θ = asin.(sinθ[:, 1])

# diffusivity
# bottom enhanced:
κ0 = 6e-5
κ1 = 2e-3
h = 200
#= # not bottom enhanced: =#
#= κ0 = 1e-4 =#
#= κ1 = 0 =#
#= h = 200 =#
κ = @. κ0 + κ1*exp(-(z + H(x))/h)

# timestepping
secsInDay = 86400
Δt = 10*secsInDay
tPlot = 100*Δt
tSave = 100*Δt

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
log(ofile, "\nPGSolver with Parameters\n")

log(ofile, @sprintf("nξ = %d", nξ))
log(ofile, @sprintf("nσ = %d\n", nσ))
log(ofile, @sprintf("L  = %d km", L/1000))
log(ofile, @sprintf("H0 = %d m", H0))
log(ofile, @sprintf("Pr = %1.1f", Pr))
log(ofile, @sprintf("f  = %1.1e s-1", f))
log(ofile, @sprintf("N  = %1.1e s-1", N))
log(ofile, @sprintf("κ0 = %1.1e m2 s-1", κ0))
log(ofile, @sprintf("κ1 = %1.1e m2 s-1", κ1))
log(ofile, @sprintf("h  = %d m", h))
log(ofile, @sprintf("Δt = %.2f days", Δt/86400))

log(ofile, string("\nVariations in ξ: ", ξVariation))

log(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*Pr*κ1/abs(f))))
log(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", H0*(σ[2] - σ[1])))
close(ofile)
