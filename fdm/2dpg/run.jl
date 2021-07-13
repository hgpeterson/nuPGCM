using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx, SpecialFunctions

plt.style.use("../../plots.mplstyle")
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
sol_U = computeSol(inversionRHS) 
println("Done.") 
b = evolve(5*tSave) 

# b = evolve(15*secsInYear; bl=true) 

################################################################################
# plots
################################################################################

path = ""
dfiles = string.(path, "checkpoint", 1:5, ".h5")
profilePlot(dfiles, argmin(abs.(ξ .- L/4))) 

#= c = loadCheckpoint2DPG("checkpoint30.h5") =#
#= ridgePlot(c.χ, c.b, "t = 30 days", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; vext=2.5e-2) =#
#= savefig("chi030.png") =#
#= close() =#
#= c = loadCheckpoint2DPG("checkpoint31.h5") =#
#= ridgePlot(c.χ, c.b, "t = 31 days", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; vext=2.5e-2) =#
#= savefig("chi031.png") =#
#= close() =#
