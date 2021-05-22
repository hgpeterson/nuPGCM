# parameters (as in RC20)
L = 2e6
H0 = 2e3
Pr = 1e0
f = -5.5e-5
N = 1e-3
β = 2e-11
r = 0.1*β*L

# topography
amp =  0.4*H0
H(x) = H0 - amp*sin(2*pi*x/L) # hill
Hx(x) = -2*pi/L*amp*cos(2*pi*x/L)

# number of grid points
nx = 2^8 + 1 
nz = 2^8

# domain in physical (x, z) space
dx = L/nx
x = 0:dx:(L - dx)
xx = repeat(x, 1, nz)
σ = @. -(cos(pi*(0:nz-1)/(nz-1)) + 1)/2 # chebyshev 
zz = repeat(σ', nx, 1).*repeat(H.(x), 1, nz)

# arrays of sin(θ) and cos(θ) 
sinθ = @. -Hx(xx)/sqrt(1 + Hx(xx)^2)
cosθ = @. 1/sqrt(1 + Hx(xx)^2) 

# domain in locally rotated (x, ẑ) space
ẑẑ = @. zz/cosθ

# diffusivity
κ0 = 6e-7
κ1 = 2e-5
h = 200
#= κ = κ1*ones(nx, nz) =#
κ = @. κ0 + κ1*exp(-(ẑẑ + H(xx))/h)

# print properties
println("\nPGSolver with Parameters\n")

println(@sprintf("nx = %d", nx))
println(@sprintf("nz = %d\n", nz))

println(@sprintf("L  = %d km", L/1000))
println(@sprintf("H0 = %d m", H0))
println(@sprintf("Pr = %1.1f", Pr))
println(@sprintf("f  = %1.1e s-1", f))
println(@sprintf("N  = %1.1e s-1", N))
println(@sprintf("β  = %1.1e m-1 s-1", β))
println(@sprintf("r  = %1.1e s-1", r))
println(@sprintf("κ0 = %1.1e m2 s-1", κ0))
println(@sprintf("κ1 = %1.1e m2 s-1", κ1))
println(@sprintf("h  = %d m", h))

println(@sprintf("\nBL thickness ~ %1.2f m", sqrt(r*N^2*Hx(x[1, 1])^2/(κ[1, 1]*(f^2 + r^2)))^-1))
println(@sprintf(" ẑ[2] - ẑ[1] at x = 0 ~ %1.2f m", (ẑẑ[1, 2] - ẑẑ[1, 1])))
