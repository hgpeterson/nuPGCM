# parameters (as in RC20)
Pr = 1e0
f = -5.5e-5
N = 1e-3
H = 2e3
# H = 3673.32793219601
θ = 2.5e-3
# θ = -0.03637523599970141

# z grid
nẑ = 2^8
z = @. -H*(cos(pi*(0:nẑ-1)/(nẑ-1)) + 1)/2 # chebyshev 

# domain in locally rotated ẑ space
ẑ = z/cos(θ)

# diffusivity
# bottom enhanced: 
κ0 = 6e-5 
κ1 = 2e-3 
h = 200 
# # not bottom enhanced:
# κ0 = 1e-4
# κ1 = 0
# h = 200
κ = @. κ0 + κ1*exp(-(z + H)/h)

# set U = U₀ or compute U at each time step?
# transportConstraint = true
transportConstraint = false
U₀ = 0

# timestepping
secsInDay = 86400
secsInYear = 360*secsInDay
Δt = 10*secsInDay
# tSave = 3*secsInYear
tSave = 60*secsInDay

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
log(ofile, "\nTransport-Contraned 1D PG Model with Parameters\n")

log(ofile, @sprintf("nẑ = %d\n", nẑ))
log(ofile, @sprintf("H  = %d m", H))
log(ofile, @sprintf("Pr = %1.1f", Pr))
log(ofile, @sprintf("f  = %1.1e s-1", f))
log(ofile, @sprintf("N  = %1.1e s-1", N))
log(ofile, @sprintf("θ  = %1.1e rad", θ))
log(ofile, @sprintf("κ0 = %1.1e m2 s-1", κ0))
log(ofile, @sprintf("κ1 = %1.1e m2 s-1", κ1))
log(ofile, @sprintf("h  = %d m", h))
log(ofile, @sprintf("Δt = %.2f days", Δt/secsInDay))

log(ofile, string("Transport Constraint:   ", transportConstraint))

log(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*Pr*κ[1]/abs(f))))
log(ofile, @sprintf("          ẑ[2] - ẑ[1] ~ %1.2f m\n", ẑ[2] - ẑ[1]))
close(ofile)