using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx

plt.style.use("C:/paper_plots.mplstyle")
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

ũ, ṽ, b̃, P̃x̃ = evolve(5*τ_A)

################################################################################
# plots
################################################################################

path = ""
iSaves = 0:1:5
dfiles = string.(path, "checkpoint", iSaves, ".h5")
profilePlot(dfiles)

#= # farfieldv grid =#
#= file = h5open("vs.h5", "r") =#
#= vs = read(file, "vs") =#
#= v0 = read(file, "ṽ_0") =#
#= τ_Ss = read(file, "τ_Ss") =#
#= τ_As = read(file, "τ_As") =#
#= close(file) =#
#= fig, ax = subplots(1) =#
#= ax.set_xlabel(L"$\tau_S$") =#
#= ax.set_ylabel(L"$\tau_A$") =#
#= img = ax.pcolormesh(τ_Ss, τ_As, vs'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1) =#
#= cb = colorbar(img, ax=ax, label=L"far-field $v/v_0$ at $t = 5\tau_A$") =#
#= ax.loglog([0, 1], [0, 1], transform=ax.transAxes, "k--", lw=0.5) =#
#= tight_layout() =#
#= savefig("farfieldv.png") =#

################################################################################
# sweep parameter space
################################################################################

#= τ_As = 10 .^range(1, 4, length=2^5) =#
#= τ_Ss = 10 .^range(1, 4, length=2^5) =#
#= Eks = 1 ./τ_Ss.^2 =#
#= Ss = 1 ./τ_As =#
#= vs = zeros((size(τ_Ss, 1), size(τ_As, 1))) =#

#= canonical = false =#
#= Pr = 1e3 =#
#= ν0 = 1 =#
#= ν1 = 0 =#
#= κ0 = 0 =#
#= κ1 = 0 =#
#= h = 10 =#
#= ṽ_0 = -1 =#
#= bottomIntense = false =#
#= adaptiveTimestep = false =#
#= α = 0.5 =#
#= for i=1:size(τ_Ss, 1) =#
#=     println("i = ", i) =#
#=     for j=1:size(τ_As, 1) =#
#=         global Ek = Eks[i] =#
#=         global S = Ss[j] =#
#=         global τ_S = τ_Ss[i] =#
#=         global τ_A = τ_As[j] =#
#=         global Δt = minimum([τ_S/100, τ_A/100]) =#
#=         global H = τ_S =#
#=         if H >= 1e3 =#
#=             global nz̃ = 2^11 =#
#=         elseif H >= 1e2 =#
#=             global nz̃ = 2^10 =#
#=         else =#
#=             global nz̃ = 2^9 =#
#=         end =#
#=         global z̃ = @. H*(1 - cos(pi*(0:nz̃-1)/(nz̃-1)))/2 =#
#=         global ν = @. ν0 + ν1*exp(-z̃/h) =#
#=         global κ = @. κ0 + κ1*exp(-z̃/h) =#

#=         global tSave = 10*τ_A =#
#=         #1= global tSave = τ_A =1# =#
        
#=         global ũ, ṽ, b̃, P̃x̃ = evolve(5*τ_A) =#
#=         vs[i, j] = ṽ[end] =#
#=         #1= profilePlot(string.("checkpoint", 0:1:5, ".h5"); fname=string("profiles_", i, "_", j, ".png")) =1# =#
#=     end =#
#= end =#

#= file = h5open("vs.h5", "w") =#
#= write(file, "vs", vs) =#
#= write(file, "ṽ_0", ṽ_0) =#
#= write(file, "τ_Ss", τ_Ss) =#
#= write(file, "τ_As", τ_As) =#
#= close(file) =#
