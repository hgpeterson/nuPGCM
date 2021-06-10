using SparseArrays, LinearAlgebra, Printf, HDF5, PyPlot, PyCall, SpecialFunctions

plt.style.use("C:/paper_plots.mplstyle")
close("all")
pygui(false)

include("../../../myJuliaLib.jl")
include("params.jl")
include("plotting.jl")
include("utils.jl")
include("inversion.jl")
include("evolution.jl")
include("steady.jl")

################################################################################
# run evolution integrations
################################################################################

print("Computing inversion matrix: ")
inversionLHS = lu(getInversionLHS())
println("Done.")

b = evolve(5*tSave)

# b = steadyState()

################################################################################
# plots
################################################################################

ii = [0, 1, 2, 3, 4, 5]
profilePlot(string.("checkpoint", ii, ".h5"))
# ii = [0, 1, 2, 3, 4, 5, 999]
# profilePlot(string.("checkpoint", ii, ".h5"))