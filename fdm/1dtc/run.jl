using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx

plt.style.use("~/paper_plots.mplstyle")
close("all")
pygui(false)

include("../../myJuliaLib.jl")
include("params.jl")
include("evolution.jl")
include("plotting.jl")
include("utils.jl")

################################################################################
# run single integration
################################################################################

û, v̂, b, Px = evolve(5*tSave)

################################################################################
# plots
################################################################################

path = ""
iSaves = 0:1:5
dfiles = string.(path, "checkpoint", iSaves, ".h5")
profilePlot(dfiles)
