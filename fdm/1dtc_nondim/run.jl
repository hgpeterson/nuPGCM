using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx

plt.style.use("../../plots.mplstyle")
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

# file = h5open("vs.h5", "r")
# vs_5A = read(file, "vs_5A")
# vs_5S = read(file, "vs_5S")
# ṽ_0 = read(file, "ṽ_0")
# τ_Ss = read(file, "τ_Ss")
# τ_As = read(file, "τ_As")
# close(file)

# aspect = log(τ_As[end]/τ_As[1])/log(τ_Ss[end]/τ_Ss[1])
# fig, ax = subplots(1, 2, figsize=(3.404, 3), sharey=true)
# ax[1].set_box_aspect(aspect)
# ax[1].set_xlabel(L"spin-down time, $\tilde{\tau}_S$")
# ax[1].set_ylabel(L"arrest time, $\tilde{\tau}_A$")
# ax[1].set_title(L"$t = 5\tilde\tau_A$")
# ax[1].spines["left"].set_visible(false)
# ax[1].spines["bottom"].set_visible(false)
# ax[1].set_xlim([τ_Ss[1], τ_Ss[end]])
# ax[1].set_ylim([τ_As[1], τ_As[end]])
# img = ax[1].pcolormesh(τ_Ss, τ_As, vs_5A'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
# cb = colorbar(img, ax=ax[:], shrink=0.63, label=L"far-field along-slope flow, $\tilde{v}/\tilde{v}_0$", orientation="horizontal")
# ax[1].loglog([1e1, 1e4], [1e1, 1e4], "w--", lw=0.5)

# ax[2].set_box_aspect(aspect)
# ax[2].set_xlabel(L"spin-down time, $\tilde{\tau}_S$")
# ax[2].set_title(L"$t = 5\tilde\tau_S$")
# ax[2].spines["left"].set_visible(false)
# ax[2].spines["bottom"].set_visible(false)
# ax[2].set_xlim([τ_Ss[1], τ_Ss[end]])
# ax[2].set_ylim([τ_As[1], τ_As[end]])
# img = ax[2].pcolormesh(τ_Ss, τ_As, vs_5S'/ṽ_0, rasterized=true, shading="auto", vmin=0, vmax=1)
# ax[2].loglog([1e1, 1e4], [1e1, 1e4], "w--", lw=0.5)

# subplots_adjust(left=0.15, right=0.85, bottom=0.35, top=0.9, wspace=0.2, hspace=0.6)

# println("spindownGrid.png")
# savefig("spindownGrid.png")
# close()

################################################################################
# sweep parameter space
################################################################################

# τ_As = 10 .^range(0, 3, length=2^7) 
# τ_Ss = 10 .^range(1, 3, length=2^7) 
# Eks = 1 ./τ_Ss.^2 
# Ss = 1 ./τ_As 
# vs_5A = zeros((size(τ_Ss, 1), size(τ_As, 1))) 
# vs_5S = zeros((size(τ_Ss, 1), size(τ_As, 1))) 

# canonical = false 
# # canonical = true 
# ν0 = 1 
# ν1 = 0 
# κ0 = 0 
# κ1 = 0 
# h = 10 
# ṽ_0 = -1 
# α = 0.5 
# N = 1
# for i=1:size(τ_Ss, 1) 
#     println("i = ", i) 
#     for j=1:size(τ_As, 1) 
#         global Ek = Eks[i] 
#         global S = Ss[j] 
#         global τ_S = τ_Ss[i] 
#         global τ_A = τ_As[j] 
#         global Δt = minimum([τ_S/100, τ_A/100]) 
#         global H = τ_S 
#         if H >= 1e3 
#             global nz̃ = 2^11 
#         elseif H >= 1e2 
#             global nz̃ = 2^10 
#         else 
#             global nz̃ = 2^9 
#         end 
#         global z̃ = @. H*(1 - cos(pi*(0:nz̃-1)/(nz̃-1)))/2 
#         global ν = @. ν0 + ν1*exp(-z̃/h) 
#         global κ = @. κ0 + κ1*exp(-z̃/h) 

#         global tSave = nothing
#         # global tSave = τ_A 
       
#         global ũ, ṽ, b̃, P̃x̃ = evolve(5*τ_A) 
#         vs_5A[i, j] = ṽ[end] 
#         global ũ, ṽ, b̃, P̃x̃ = evolve(5*τ_S) 
#         vs_5S[i, j] = ṽ[end] 
#         # profilePlot(string.("checkpoint", 0:1:5, ".h5"); fname=string("profiles_", i, "_", j, ".png"))
#     end 
# end 

# file = h5open("vs.h5", "w") 
# write(file, "vs_5A", vs_5A) 
# write(file, "vs_5S", vs_5S) 
# write(file, "ṽ_0", ṽ_0) 
# write(file, "τ_Ss", τ_Ss) 
# write(file, "τ_As", τ_As) 
# close(file) 