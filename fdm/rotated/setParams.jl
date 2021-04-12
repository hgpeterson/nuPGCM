# parameters (as in RC20)
Pr = 1e0
f = -5.5e-5
N = 1e-3

# set U = 0 or compute U at each time step?
#= symmetry = false =#
symmetry = true

# topography
L = 2e6
H0 = 2e3
amp =  0.4*H0
H(x) = H0 - amp*sin(2*pi*x/L - π/2) 
Hx(x) = -2*pi/L*amp*cos(2*pi*x/L - π/2)

# gridpoints 
nx = 1
#= nx = 2^8 + 1 =# 
nz = 2^8

# x grid
x = repeat([L/4], 1, nz)
#= dx = L/nx =#
#= x = repeat(0:dx:(L - dx), 1, nz) =#

# z grid
σ = @. -(cos(pi*(0:nz-1)/(nz-1)) + 1)/2 # chebyshev 
z = repeat(σ', nx, 1).*repeat(H.(x[:, 1]), 1, nz)

# arrays of sin(θ) and cos(θ) 
sinθ = @. -Hx(x)/sqrt(1 + Hx(x)^2)
cosθ = @. 1/sqrt(1 + Hx(x)^2) 
θ = asin.(sinθ[:, 1])

# domain in locally rotated (x, ẑ) space
ẑ = @. z/cosθ

# diffusivity
κ0 = 6e-5
κ1 = 2e-3
h = 200
#= bottomIntense = true =#
bottomIntense = false
if bottomIntense
    κ = @. κ0 + κ1*exp(-(ẑ + H(x))/h)
else
    κ1 = 1e-4
    κ = κ1*ones(nx, nz)
end

# timestepping
Δt = 10*86400
adaptiveTimestep = false

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
log(ofile, "\nRotated PGSolver with Parameters\n")

log(ofile, @sprintf("nx = %d", nx))
log(ofile, @sprintf("nz = %d\n", nz))
log(ofile, @sprintf("L  = %d km", L/1000))
log(ofile, @sprintf("H0 = %d m", H0))
log(ofile, @sprintf("Pr = %1.1f", Pr))
log(ofile, @sprintf("f  = %1.1e s-1", f))
log(ofile, @sprintf("N  = %1.1e s-1", N))
log(ofile, @sprintf("κ0 = %1.1e m2 s-1", κ0))
log(ofile, @sprintf("κ1 = %1.1e m2 s-1", κ1))
log(ofile, @sprintf("h  = %d m", h))
log(ofile, @sprintf("Δt = %.2f days", Δt/86400))

log(ofile, string("Symmetric:              ", symmetry))
log(ofile, string("Bottom intensification: ", bottomIntense))
log(ofile, string("Adaptive timestep:      ", adaptiveTimestep))

log(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*Pr*κ1/abs(f))))
log(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", H0*(σ[2] - σ[1])))
close(ofile)
