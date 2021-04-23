using SparseArrays, LinearAlgebra, Printf, HDF5, PyPlot, PyCall

plt.style.use("~/paper_plots.mplstyle")
close("all")
pygui(false)

include("../../myJuliaLib.jl")
include("setParams.jl")
include("plottingLib.jl")
include("rotated.jl")
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

b = steadyState()

################################################################################
# plots
################################################################################

#= ii = [1, 2, 3, 4, 5, 999] =#
#= profilePlot(string.("checkpoint", ii, ".h5")) =#

#= include("talkPlots.jl") =#
#= vAnimation("images/constKappa/") =#
#= uProfile("images/constKappa/") =#
#= uProfile("images/biKappa/") =#
