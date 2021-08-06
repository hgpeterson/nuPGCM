################################################################################
# Run 2D νPGCM
################################################################################

using PyPlot, PyCall, Printf, SparseArrays, SuiteSparse, LinearAlgebra, HDF5, Dierckx, SpecialFunctions

# plotting stylesheet
plt.style.use("../../plots.mplstyle")
close("all")
pygui(false)

# libraries
include("../../myJuliaLib.jl")
include("setup.jl")
include("utils.jl")
include("plotting.jl")
include("inversion.jl")
include("evolution.jl")

# constants
const secsInDay = 86400
const secsInYear = 360*86400

"""
    run(; bl=false)

Run a simulation.
"""
function run(; bl = false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    N = 1e-3
    ξVariation = true
    L = 2e6
    # L = 1e5
    nξ = 2^8 + 1 
    nσ = 2^8

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(0:L/nξ:(L - L/nξ))
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: sine
    global symmetry = true
    H0 = 2e3
    amp =  0.4*H0
    H_func(x) = H0 - amp*sin(2*π*x/L - π/2)
    Hx_func(x) = -2*π/L*amp*cos(2*π*x/L - π/2)

    # # topography: skew gaussian
    # global const symmetry = false
    # L = 2e6 
    # H0 = 2.5e3 
    # amp = 0.65*H0 
    # ϕ(s) = exp(-s^2/2) 
    # Φ(s) = 1/2*(1 + erf(s/√2)) 
    # α = 3 
    # μ = L/3 
    # ω = L/5 
    # H_func(x) = H0 - amp*ϕ((x - μ)/ω)*Φ(α*(x - μ)/ω) 
    # Hx_func(x) = -amp/ω*(α/sqrt(2π)*ϕ(α*√2*(x - μ)/ω)*ϕ((x - μ)/ω) - (x - μ)/ω*ϕ((x - μ)/ω)*Φ(α*(x - μ)/ω)) 
    
    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    Pr = 1e0
    ν_func(ξ, σ) = Pr*κ_func(ξ, σ)
    
    # timestepping
    Δt = 10*secsInDay
    # Δt = 1*secsInDay
    tPlot = 3*secsInYear
    tSave = 3*secsInYear
    
    # create model struct
    m = ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, H_func, Hx_func, ν_func, κ_func, Δt)

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
    evolve!(m, s, 5*tSave, tPlot, tSave; bl=bl) 

    return m, s
end

m, s = run(; bl=true)
# m, s = run(bl=false)

################################################################################
# plots
################################################################################

path = ""
setupFile = "setup.h5"
m = loadSetup2DPG(setupFile)
stateFiles = string.(path, "checkpoint", 1:5, ".h5")
profilePlot(setupFile, stateFiles, argmin(abs.(m.ξ .- m.L/4))) 