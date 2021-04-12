using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx, SpecialFunctions

plt.style.use("~/paper_plots.mplstyle")
close("all")
pygui(false)

include("../myJuliaLib.jl")
include("setParams.jl")
include("terrainFollowing.jl")
include("plottingLib.jl")
include("inversion.jl")
include("evolution.jl")

################################################################################
# run evolution integrations
################################################################################

#= print("Computing inversion matrices: ") =#
#= inversionLHSs = Array{Any}(undef, nξ) =#
#= for i=1:nξ =#
#=     inversionLHSs[i] = lu(getInversionLHS(κ[i, :], H(ξ[i]))) =#
#= end =# 
#= # particular solution =# 
#= inversionRHS = getInversionRHS(zeros(nξ, nσ), 1) =#
#= solᵖ = computeSol(inversionRHS) =#
#= println("Done.") =#

#= b = evolve(5000) =#

################################################################################
# plots
################################################################################

#= path = "" =#
#= dfiles = string.(path, ["checkpoint1000.h5", "checkpoint2000.h5", "checkpoint3000.h5", "checkpoint4000.h5", "checkpoint5000.h5"]) =#
#= profilePlot(dfiles, argmin(abs.(ξ .- L/4))) =#

include("talkPlots.jl")
path = "/home/hpeter/ResearchCallies/sims/" 
#= chi_v_ridge(string(path, "sim021/")) =#
#= profiles2Dvs1D(string(path, "sim021/")) =#
#= spindownProfiles(string(path, "sim022/tauA1e2_tauS5e3/")) # ratio small =#
#= spindownProfiles(string(path, "sim022/tauA1e2_tauS1e2/")) # ratio big =#
#= spindownGrid(string(path, "sim022/")) =#
#= asymmetricRidge(string(path, "sim020/")) =#
chiForSketch(string(path, "sim023/"))
#= sketchRidge() =#
#= sketchSlope() =#
