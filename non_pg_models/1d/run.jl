################################################################################
# Run 1D PG Model
################################################################################

# run setup
include("setup.jl")

# plotting stylesheet
using PyPlot
plt.style.use("../plots.mplstyle")
close("all")
pygui(false)

function run()
    # parameters (see `setup.jl`)
    f = -5.5e-5
    N2 = 1e-6
    nz = 2^8
    # H = 2e3
    H = 1e3
    θ = 2.5e-3
    # θ = atan(sqrt(0.5*f^2/N2))   # S = 0.5
    # θ = atan(sqrt(0.001*f^2/N2)) # S = 0.001
    transportConstraint = true
    U = [0.0]
    Uamp = 0.0
    # Uamp = 1e0
    Uper = secsInDay/2

    # grid: chebyshev
    z = @. -H*(cos(pi*(0:nz-1)/(nz-1)) + 1)/2 # chebyshev 
    
    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(z) = κ0 + κ1*exp(-(z + H)/h)
    κ_z_func(z) = -κ1/h*exp(-(z + H)/h)

    # viscosity
    μ = 1e0
    ν_func(z) = μ*κ_func(z)
    
    # timestepping
    # Δt = secsInDay/8
    Δt = 1*secsInDay
    tSave = 1000*secsInDay
    # tSave = 3*secsInYear
    # tSave = Δt
    
    # create model struct
    m = ModelSetup1D(f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transportConstraint, U, Uamp, Uper)

    # save and log params
    saveSetup1D(m)

    # set initial state
    b = zeros(nz)
    u = zeros(nz)
    v = zeros(nz)
    ∂ₓP = [0]
    i = [1]
    s = ModelState1D(b, u, v, ∂ₓP, i)

    # solve transient
    evolve!(m, s, 5000*secsInDay, tSave)
    # evolve!(m, s, 15*secsInYear, tSave)
    # evolve!(m, s, 100*Δt, tSave)
    
    # solve steady state
    # steadyState(m)

    return m, s
end

################################################################################
# run
################################################################################

m, s = run()

################################################################################
# plots
################################################################################

# setupFile = string(outFolder, "setup.h5")
# # stateFiles = string.(outFolder, "state", -1:5, ".h5")
# stateFiles = string.(outFolder, "state", 0:5, ".h5")
# profilePlot(setupFile, stateFiles)

println("Done.")