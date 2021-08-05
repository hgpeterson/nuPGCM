using PyPlot, PyCall, Printf, SparseArrays, SuiteSparse, LinearAlgebra, HDF5, Dierckx, SpecialFunctions

# plotting stylesheet
plt.style.use("../../plots.mplstyle")
close("all")
pygui(false)

# libraries
include("../../myJuliaLib.jl")
include("structs.jl")
include("utils.jl")
include("plotting.jl")
include("inversion.jl")
include("evolution.jl")

# constants
const secsInDay = 86400
const secsInYear = 360*86400
const symmetry = true

"""
    run()

Run a simulation.
"""
function run()
    # parameters (see `structs.jl`)
    f = -5.5e-5
    N = 1e-3
    ξVariation = true
    nξ = 2^8 + 1 
    nσ = 2^8
    
    # topography: sine
    # L = 2e6
    L = 1e5
    H0 = 2e3
    amp =  0.4*H0
    H_func(x) = H0 - amp*sin(2*π*x/L - π/2)
    Hx_func(x) = -2*π/L*amp*cos(2*π*x/L - π/2)
    
    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    Pr = 1e0
    ν_func(ξ, σ) = Pr*κ_func(ξ, σ)
    
    # timestepping
    # Δt = 10*secsInDay
    Δt = 1*secsInDay
    tPlot = 3*secsInYear
    tSave = 3*secsInYear
    
    # create model struct
    m = ModelSetup(f, N, ξVariation, L, nξ, nσ, H_func, Hx_func, ν_func, κ_func, Δt)

    # save and log params
    saveSetup2DPG(m)
    ofile = open("out.txt", "w")
    logParams(ofile, "\n2D νPGCM with Parameters\n")
    logParams(ofile, @sprintf("nξ = %d", nξ))
    logParams(ofile, @sprintf("nσ = %d\n", nσ))
    logParams(ofile, @sprintf("L  = %d km", L/1000))
    logParams(ofile, @sprintf("H0 = %d m", H0))
    logParams(ofile, @sprintf("Pr = %1.1f", Pr))
    logParams(ofile, @sprintf("f  = %1.1e s-1", f))
    logParams(ofile, @sprintf("N  = %1.1e s-1", N))
    logParams(ofile, @sprintf("κ0 = %1.1e m2 s-1", κ0))
    logParams(ofile, @sprintf("κ1 = %1.1e m2 s-1", κ1))
    logParams(ofile, @sprintf("h  = %d m", h))
    logParams(ofile, @sprintf("Δt = %.2f days", Δt/86400))
    logParams(ofile, string("\nVariations in ξ: ", ξVariation))
    logParams(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*Pr*κ1/abs(f))))
    logParams(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", H0*(m.σ[2] - m.σ[1])))
    close(ofile)

    # set initial state
    b = N^2*m.z
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState(b, χ, uξ, uη, uσ, i)

    # solve
    # evolve!(m, s, 5*tSave, tPlot, tSave) 
    evolve!(m, s, 5*tSave, tPlot, tSave; bl=true) 

    return m, s
end

m, s = run()

################################################################################
# plots
################################################################################

path = ""
setupFile = "setup.h5"
m = loadSetup2DPG(setupFile)
stateFiles = string.(path, "checkpoint", 1:5, ".h5")
profilePlot(setupFile, stateFiles, argmin(abs.(m.ξ .- m.L/4))) 
