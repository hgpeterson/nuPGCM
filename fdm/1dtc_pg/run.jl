using SparseArrays, LinearAlgebra, Printf, HDF5, PyPlot, PyCall

plt.style.use("../../plots.mplstyle")
close("all")
pygui(false)

include("../../myJuliaLib.jl")
include("params.jl")
include("plotting.jl")
include("utils.jl")
include("inversion.jl")
include("evolution.jl")
include("steady.jl")

################################################################################
# Setup matrices 
################################################################################

#= print("Computing inversion matrix: ") =#
#= inversionLHS = lu(getInversionLHS()) =#
#= println("Done.") =#

################################################################################
# run evolution integrations
################################################################################

#= b = evolve(5*tSave) =#
#= b = evolveBL(5*tSave) =#

#= b = steadyState() =#

################################################################################
# plots
################################################################################

#= ii = [0, 1, 2, 3, 4, 5] =#
#= profilePlot(string.("checkpoint", ii, ".h5")) =#
#= ii = [0, 1, 2, 3, 4, 5, 999] =#
#= profilePlot(string.("checkpoint", ii, ".h5")) =#

ii = [0, 1, 2, 3, 4, 5]
path = "/home/hpeter/Documents/ResearchCallies/rapid_adjustment/sims/sim028/tht2.5e-3/"
#= path = "/home/hpeter/Documents/ResearchCallies/rapid_adjustment/sims/sim028/tht2.5e-2/" =#
#= path = "/home/hpeter/Documents/ResearchCallies/rapid_adjustment/sims/sim028/tht6e-2/" =#
#= path = "/home/hpeter/Documents/ResearchCallies/rapid_adjustment/sims/sim028/Pr2e2/" =#
datafilesFull = string.(path, "full/checkpoint", ii, ".h5")
datafilesBL = string.(path, "bl/checkpoint", ii, ".h5")
profilePlotBL(datafilesFull, datafilesBL)

#= datafile = "/home/hpeter/Documents/ResearchCallies/rapid_adjustment/sims/sim028/tht6e-2/bl/checkpoint1.h5" =#
#= datafile = "checkpoint5.h5" =#
#= buoyancyFlux(datafile) =#
