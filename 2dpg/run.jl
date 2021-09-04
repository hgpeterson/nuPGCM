################################################################################
# Run 2D νPGCM
################################################################################

using PyPlot, PyCall, SpecialFunctions

# plotting stylesheet
plt.style.use("../plots.mplstyle")
close("all")
pygui(false)

# libraries
include("setup.jl")
include("plotting.jl")

# global constants
const secsInDay = 86400
const secsInYear = 360*86400
const outFolder = "out/"

function runRidge(; bl = false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    N = 1e-3
    ξVariation = true
    L = 2e6
    nξ = 2^8 + 1 
    nσ = 2^8
    coords = "cartesian"
    periodic = true

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
    H_func(x) = H0 + amp*cos(2*π*x/L)
    Hx_func(x) = -2*π/L*amp*sin(2*π*x/L)

    # # topography: skew gaussian
    # global symmetry = false
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
    # tPlot = 3*secsInYear
    tPlot = 15*secsInYear
    # tSave = 3*secsInYear
    tSave = 60*secsInDay
    
    # create model struct
    m = ModelSetup2DPG(f, N, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, Δt)

    # save and log params
    saveSetup2DPG(m)

    # set initial state
    b = N^2*m.z
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, 15*secsInYear, tPlot, tSave; bl=bl) 

    return m, s
end

function runSeamount(; bl = false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    N = 1e-3
    ξVariation = true
    L = 2e4
    # L = 2e5
    nξ = 2^6 + 1 
    nσ = 2^6
    coords = "cylindrical"
    periodic = false

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L) # avoid r = 0
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography
    global symmetry = true
    H0 = 5.5e3
    H1 = 3e3
    L0 = L/4
    H_func(x) = H0 - H1*exp(-x^2/(2*L0^2))
    Hx_func(x) = H1*x/L0^2*exp(-x^2/(2*L0^2))
    
    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    Pr = 1e0
    ν_func(ξ, σ) = Pr*κ_func(ξ, σ)
    
    # timestepping
    # Δt = 1*secsInDay
    # Δt = 60
    Δt = 3600
    tPlot = 120*secsInDay
    tSave = 1*secsInYear
    
    # create model struct
    m = ModelSetup2DPG(f, N, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, Δt)

    # save and log params
    saveSetup2DPG(m)

    # set initial state
    b = N^2*m.z
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # debug: what's the max Burger number?
    S = @. m.N^2/m.f^2*m.Hx^2
    println(maximum(S))

    # solve
    evolve!(m, s, 1*tSave, tPlot, tSave; bl=bl) 

    return m, s
end

m, s = runRidge()
# m, s = runRidge(; bl=true)
# m, s = runSeamount()
# m, s = runSeamount(; bl=true)

################################################################################
# plots
################################################################################

# setupFile = string(outFolder, "setup.h5")
# m = loadSetup2DPG(setupFile)
# stateFiles = string.(outFolder, "state", 1:5, ".h5")
# profilePlot(setupFile, stateFiles, argmin(abs.(m.ξ .- m.L/4))) 

println("Done.")