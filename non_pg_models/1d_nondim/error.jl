# This file is just to compute errors between canonical and TC solutions

using nuPGCM, PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, Dierckx

plt.style.use("../../plots.mplstyle")
close("all")
pygui(false)

include("params.jl")
include("evolution.jl")
include("plotting.jl")
include("utils.jl")

################################################################################
# sweep parameter space
################################################################################

# τ_As = 10 .^range(0, 3, length=2^3) 
# τ_Ss = 10 .^range(1, 3, length=2^3) 
# Eks = 1 ./τ_Ss.^2 
# Ss = 1 ./τ_As 
# times = [1e-2, 1e-1, 5e-1, 1, 5]
# errors = zeros(size(times, 1), size(τ_Ss, 1), size(τ_As, 1)) 

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

# 	    for k=1:size(times, 1)
#        	    global canonical = true
#             global ũ_c, ṽ_c, b̃_c, P̃x̃_c = evolve(times[k]*τ_S) 
#        	    global canonical = false
#             global ũ, ṽ, b̃, P̃x̃ = evolve(times[k]*τ_S) 
#             errors[k, i, j] = 1/H * trapz(abs.(ṽ .- ṽ_c), z̃)
# 	    end
#     end 
# end 

# file = h5open("errors.h5", "w") 
# write(file, "errors", errors) 
# write(file, "ṽ_0", ṽ_0) 
# write(file, "τ_Ss", τ_Ss) 
# write(file, "τ_As", τ_As) 
# write(file, "times", times) 
# close(file) 

################################################################################
# plot
################################################################################

file = h5open("errors.h5", "r")
errors = read(file, "errors")
ṽ_0 = read(file, "ṽ_0")
τ_Ss = read(file, "τ_Ss")
τ_As = read(file, "τ_As")
times = read(file, "times")
close(file)

for k ∈ eachindex(times)
    aspect = log(τ_As[end]/τ_As[1])/log(τ_Ss[end]/τ_Ss[1])
    fig, ax = subplots(1, figsize=(3.404, 3.404))
    ax.set_box_aspect(aspect)
    ax.set_xlabel(L"spin-down time, $\tilde{\tau}_S$")
    ax.set_ylabel(L"arrest time, $\tilde{\tau}_A$")
    ax.set_title(string(L"$t = $", @sprintf("%.2f", times[k]), L"$\tilde\tau_S$"))
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xlim([τ_Ss[1], τ_Ss[end]])
    ax.set_ylim([τ_As[1], τ_As[end]])
    img = ax.pcolormesh(τ_Ss, τ_As, errors[k, :, :]', rasterized=true, shading="auto", vmin=0, vmax=0.2)
    cb = colorbar(img, ax=ax, shrink=0.63, label=L"average error, $\frac{1}{H} \int | \tilde v_c - \tilde v |$ d$\tilde z$", orientation="horizontal", extend="max")
    ax.loglog([1e1, 1e3], [1e1, 1e3], "w--", lw=0.5)
    subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    println(@sprintf("errors%d.png", k))
    savefig(@sprintf("errors%d.png", k))
    plt.close()
end

