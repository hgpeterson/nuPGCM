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

function run(; bl = false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    nz = 2^8
    H = 2e3
    # H = 4e3
    # θ = 2.5e-3
    θ = 2.5e-2
    # H = 3673.32793219601
    # θ = -0.03639128788776821
    transportConstraint = true
    # transportConstraint = false
    U = [0.0]
    # Uamp = 1e-3
    Uamp = 1e-2
    Uper = secsInDay/2

    # grid: chebyshev unless bl
    if bl
        z = collect(-H:H/(nz-1):0) # uniform
    else
        z = @. -H*(cos(pi*(0:nz-1)/(nz-1)) + 1)/2 # chebyshev 
    end
    
    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(z) = κ0 + κ1*exp(-(z + H)/h)
    κ_z_func(z) = -κ1/h*exp(-(z + H)/h)

    # viscosity
    μ = 1e0
    ν_func(z) = μ*κ_func(z)

    # stratification
    N2 = 1e-6
    
    # timestepping
    # Δt = 1*secsInDay
    # tSave = 3*secsInYear
    Δt = secsInDay/24
    tSave = Δt
    # Δt = secsInDay/8
    # tSave = 3*secsInYear
    
    # create model struct
    m = ModelSetup1DPG(f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transportConstraint, U, Uamp, Uper)

    # save and log params
    saveSetup1DPG(m)

    # set initial state
    s = loadState1DPG(string(outFolder, "fullSim/state4.h5"))
    s.i[1] = 1
    # i = s.i
    # b = zeros(nz)
    # χ, u, v = invert(m, b)
    # i = [1]
    # s = ModelState1DPG(b, χ, u, v, i)

    # solve transient
    # evolve!(m, s, 15*secsInYear, tSave; bl=bl) 
    evolve!(m, s, 4*secsInDay, tSave; bl=bl) 
    
    # solve steady state
    # steadyState(m)

    return m, s
end

################################################################################
# run
################################################################################

# m, s = run()
# m, s = run(; bl=true)

################################################################################
# plots
################################################################################

# setupFile = string(outFolder, "setup.h5")
# # stateFiles = string.(outFolder, "state", -1:5, ".h5")
# stateFiles = string.(outFolder, "state", 0:5, ".h5")
# profilePlot(setupFile, stateFiles)

setupFile = string(outFolder, "setup.h5")
m = loadSetup1DPG(setupFile)
for i=1:96
    stateFile = @sprintf("%sstate%d.h5", outFolder, i)
    imgFile = @sprintf("%sprofiles_%03d.png", outFolder, i)
    profilePlot(m, stateFile, imgFile)
end

println("Done.")