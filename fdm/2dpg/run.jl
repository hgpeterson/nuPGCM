using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx, SpecialFunctions

plt.style.use("~/paper_plots.mplstyle")
close("all")
pygui(false)

include("../../myJuliaLib.jl")
include("params.jl")
include("utils.jl")
include("plotting.jl")
include("inversion.jl")
include("evolution.jl")

################################################################################
# run evolution integrations
################################################################################

print("Computing inversion matrices: ")
inversionLHSs = Array{Any}(undef, nξ)
for i=1:nξ
    inversionLHSs[i] = lu(getInversionLHS(κ[i, :], H(ξ[i])))
end 
# particular solution 
inversionRHS = getInversionRHS(zeros(nξ, nσ), 1)
solᵖ = computeSol(inversionRHS)
println("Done.")

b = evolve(5*tSave)

################################################################################
# plots
################################################################################

path = ""
dfiles = string.(path, "checkpoint", 1:5, ".h5")
profilePlot(dfiles, argmin(abs.(ξ .- L/4)))
