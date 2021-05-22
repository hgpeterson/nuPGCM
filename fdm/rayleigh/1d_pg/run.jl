using SparseArrays, LinearAlgebra, Printf, HDF5, PyPlot, PyCall

plt.style.use("~/paper_plots.mplstyle")
close("all")
pygui(false)

include("../../../myJuliaLib.jl")
include("setParams1D.jl")
include("plottingLib1D.jl")
include("inversion1D.jl")
include("evolution1D.jl")
include("canonical1D.jl")

################################################################################
# run evolution integrations
################################################################################
#= b = evolve(500) =#
#= profilePlot(["b1000.h5", "b2000.h5", "b3000.h5", "b4000.h5", "b5000.h5"], 1) =#

#= b = evolveCanonical1D(40000) =#
b = steadyState()
#= profilePlot(["b1000.h5", "b2000.h5", "b3000.h5", "b4000.h5", "b5000.h5"], 1) =#
