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

print("Computing inversion matrices: ")
inversionLHSs = Array{Any}(undef, nx)
for i=1:nx
    inversionLHSs[i] = lu(getInversionLHS(κ[i, :], ẑ[i, :], θ[i]))
end 
println("Done.")

################################################################################
# run evolution integrations
################################################################################

b = evolve(5000)

################################################################################
# plots
################################################################################

profilePlot(["checkpoint1000.h5", "checkpoint2000.h5", "checkpoint3000.h5", "checkpoint4000.h5", "checkpoint5000.h5"], 1)

#= include("talkPlots.jl") =#
#= vAnimation("images/constKappa/") =#
#= uProfile("images/constKappa/") =#
#= uProfile("images/biKappa/") =#
