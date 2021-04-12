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

################################################################################
# Setup matrices 
################################################################################

print("Computing inversion matrix: ")
inversionLHS = lu(getInversionLHS())
println("Done.")

################################################################################
# run evolution integrations
################################################################################

b = evolve(5*tSave)

#= b = steadyState() =#

################################################################################
# plots
################################################################################

profilePlot(string.("checkpoint", 0:5, ".h5"))

#= include("talkPlots.jl") =#
#= vAnimation("images/constKappa/") =#
#= uProfile("images/constKappa/") =#
#= uProfile("images/biKappa/") =#
