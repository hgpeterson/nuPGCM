using PyPlot, PyCall, Printf, SparseArrays, LinearAlgebra, HDF5, SpecialFunctions

plt.style.use("C:/paper_plots.mplstyle")
close("all")
pygui(false)

include("../../../myJuliaLib.jl")
include("params.jl")
include("utils.jl")
include("plotting.jl")
include("inversion.jl")
include("evolution.jl")

################################################################################
# run evolution integrations
################################################################################

# print("Computing inversion matrices: ")
# inversionLHSs = Array{Any}(undef, nξ)
# for i=1:nξ
#     inversionLHSs[i] = lu(getInversionLHS(κ[i, :], H(ξ[i])))
# end 
# # particular solution 
# inversionRHS = getInversionRHS(zeros(nξ, nσ), 1)
# sol_U = computeSol(inversionRHS)
# println("Done.")

# b = evolve(5*tSave)

################################################################################
# make some plots
################################################################################

path = ""
dfiles = string.(path, "checkpoint", 1:5, ".h5")
profilePlot(dfiles, argmin(abs.(ξ .- L/4)))

#= # bξ vs bσ =#
#= for bfile in string.("images/constKappa/full2D/", ["b1000.h5", "b2000.h5", "b3000.h5", "b4000.h5", "b5000.h5"]) =#
#=     file = h5open(bfile, "r") =#
#=     b = read(file, "b") =#
#=     t = read(file, "t") =#
#=     close(file) =#

#=     ridgePlot(ξDerivativeTF(b), b, @sprintf("t = %d days", t/86400), L"$b_\xi$") =#
#=     savefig(@sprintf("b_xi%d.png", t/86400)) =#
#=     ridgePlot(σσ.*Hx.(ξξ)./H.(ξξ).*σDerivativeTF(b), b, @sprintf("t = %d days", t/86400), L"$\sigma H_xb_\sigma/H$") =#
#=     savefig(@sprintf("b_sig%d.png", t/86400)) =#
#= end =#

#= # px =#
#= inversionLHS = lu(getInversionLHS()) =#
#= folder = "images/constKappa/full2D/" =#
#= for bfile in string.(folder, ["b1000.h5", "b2000.h5", "b3000.h5", "b4000.h5", "b5000.h5"]) =#
#=     file = h5open(bfile, "r") =#
#=     b = read(file, "b") =#
#=     t = read(file, "t") =#
#=     close(file) =#

#=     chi, uξ, uη, uσ, U = invert(b, inversionLHS) =# 

#=     u, v, w = transformFromTF(uξ, uη, uσ) =#

#=     px = f*v + r*u =#

#=     ridgePlot(px, b, @sprintf("t = %d days", t/86400), L"$p_x$") =#
#=     savefig(@sprintf("p_x%d.png", t/86400)) =#
#= end =#

#= # bx =#
#= folder = "images/constKappa/full2D/" =#
#= for bfile in string.(folder, ["b1000.h5", "b2000.h5", "b3000.h5", "b4000.h5", "b5000.h5"]) =#
#=     file = h5open(bfile, "r") =#
#=     b = read(file, "b") =#
#=     t = read(file, "t") =#
#=     close(file) =#

#=     bx = xDerivativeTF(b) =#

#=     #1= ridgePlot(bx, b, @sprintf("t = %d days", t/86400), L"$b_x$") =1# =#
#=     ridgePlot(bx, b, @sprintf("t = %d days", t/86400), L"$b_x$"; vext=1e-4*maximum(abs.(bx))) =#
#=     savefig(@sprintf("bx%d.png", t/86400)) =#
#= end =#

#= # test inversion =#
#= nPts = nξ*nσ =#
#= umap = reshape(1:nPts, nξ, nσ) =#    
#= iU = (nPts+1):(nPts+1+nξ) # add nξ equations for vertically integrated zonal flow =#
#= rhsVec = zeros(nPts + nξ) =#                      

#= # bx = -1 in BL, 0 above =#
#= rhs = zeros(nξ, nσ) =#
#= rhs[z .<= -H.(x).+400] .= -1e-10 =#
#= rhsVec[umap[:, :]] = reshape(rhs, nPts, 1) =#
#= rhsVec[umap[:, [1, nσ]]] .= 0 # boundary conditions require zeros on the rhs =#

#= # invert =#
#= inversionLHS = getInversionLHS() =#
#= sol = inversionLHS\rhsVec =#
#= chi, uξ, uη, uσ, U = postProcess(sol) =#

#= # plot =#
#= ridgePlot(rhs, zeros(nξ, nσ), "", L"$b_x$") =#
#= savefig("bx.png") =#
#= ridgePlot(chi, zeros(nξ, nσ), "", L"$\chi$") =#
#= savefig("chi.png") =#

#= fig, ax = subplots(1, 2, sharey=true) =#
#= ax[1].plot(rhs[1, :], z[1, :]) =#
#= ax[2].plot(chi[1, :], z[1, :]) =#
#= ax[1].set_xlabel(L"$b_x$") =#
#= ax[2].set_xlabel(L"$\chi$") =#
#= tight_layout() =#
#= savefig("profiles.png") =#

#= # advection terms =#
#= inversionLHS = lu(getInversionLHS()) =#
#= figP, axP = subplots(2, 3, figsize=(6.5, 6.5/1.8), sharey=true) =#
#= folder = "images/constKappa/full2D/" =#
#= for bfile in string.(folder, ["b1000.h5", "b2000.h5", "b3000.h5", "b4000.h5", "b5000.h5"]) =#
#=     file = h5open(bfile, "r") =#
#=     b = read(file, "b") =#
#=     t = read(file, "t") =#
#=     close(file) =#

#=     chi, uξ, uη, uσ, U = invert(b, inversionLHS) =# 

#=     adv1 = -uξ.*ξDerivativeTF(b) =#
#=     adv2 = -uσ.*σDerivativeTF(b) =#
#=     adv3 = -N^2*uξ.*Hx.(ξξ).*σσ =#
#=     adv4 = -N^2*uσ.*H.(ξξ) =#
#=     diff = σDerivativeTF(κ.*(N^2 .+ σDerivativeTF(b)./H.(ξξ)))./H.(ξξ) =#
#=     sum = adv1 + adv2 + adv3 + adv4 + diff =#

#=     fig, ax = subplots(2, 3, figsize=(6.5, 6.5/1.8), sharey=true, sharex=true) =#
#=     img = ax[1, 1].pcolormesh(ξξ/1000, σσ, adv1, vmin=-maximum(abs.(adv1)), vmax=maximum(abs.(adv1)), cmap="RdBu_r") =#
#=     colorbar(img, ax=ax[1, 1], label=L"-u^\xi b_\xi") =#
#=     img = ax[1, 2].pcolormesh(ξξ/1000, σσ, adv2, vmin=-maximum(abs.(adv2)), vmax=maximum(abs.(adv2)), cmap="RdBu_r") =#
#=     colorbar(img, ax=ax[1, 2], label=L"-u^\sigma b_\sigma") =#
#=     img = ax[1, 3].pcolormesh(ξξ/1000, σσ, adv3, vmin=-maximum(abs.(adv3)), vmax=maximum(abs.(adv3)), cmap="RdBu_r") =#
#=     colorbar(img, ax=ax[1, 3], label=L"-N^2u^\xi H_x\sigma") =#
#=     img = ax[2, 1].pcolormesh(ξξ/1000, σσ, adv4, vmin=-maximum(abs.(adv4)), vmax=maximum(abs.(adv4)), cmap="RdBu_r") =#
#=     colorbar(img, ax=ax[2, 1], label=L"-N^2u^\sigma H") =#
#=     img = ax[2, 2].pcolormesh(ξξ/1000, σσ, diff, vmin=-maximum(abs.(diff)), vmax=maximum(abs.(diff)), cmap="RdBu_r") =#
#=     cb = colorbar(img, ax=ax[2, 2], label=L"H^{-1}[\kappa(N^2 + H^{-1}b_\sigma)]_\sigma") =#
#=     cb.ax.ticklabel_format(style="sci", scilimits=(-3, 3)) =#
#=     img = ax[2, 3].pcolormesh(ξξ/1000, σσ, sum, vmin=-maximum(abs.(sum)), vmax=maximum(abs.(sum)), cmap="RdBu_r") =#
#=     cb = colorbar(img, ax=ax[2, 3], label=L"b_t") =#
#=     cb.ax.ticklabel_format(style="sci", scilimits=(-3, 3)) =#
#=     ax[1, 1].set_ylabel(L"\sigma") =#
#=     ax[2, 1].set_ylabel(L"\sigma") =#
#=     ax[2, 1].set_xlabel(L"\xi") =#
#=     ax[2, 2].set_xlabel(L"\xi") =#
#=     ax[2, 3].set_xlabel(L"\xi") =#
#=     fig.tight_layout() =#
#=     fig.savefig(@sprintf("evol%d.png", t/86400)) =#

#=     axP[1, 1].plot(adv1[1, :], σ, label=string("Day ", Int64(t/86400))) =#
#=     axP[1, 1].set_xlabel(L"-u^\xi b_\xi") =#
#=     axP[1, 2].plot(adv2[1, :], σ) =#
#=     axP[1, 2].set_xlabel(L"-u^\sigma b_\sigma") =#
#=     axP[1, 3].plot(adv3[1, :], σ) =#
#=     axP[1, 3].set_xlabel(L"-N^2u^\xi H_x\sigma") =#
#=     axP[2, 1].plot(adv4[1, :], σ) =#
#=     axP[2, 1].set_xlabel(L"-N^2u^\sigma H") =#
#=     axP[2, 2].plot(diff[1, :], σ) =#
#=     axP[2, 2].set_xlabel(L"H^{-1}[\kappa(N^2 + H^{-1}b_\sigma)]_\sigma") =#
#=     axP[2, 3].plot(sum[1, :], σ) =#
#=     axP[2, 3].set_xlabel(L"b_t") =#
#= end =#
#= axP[1, 1].set_ylabel(L"\sigma") =#
#= axP[2, 1].set_ylabel(L"\sigma") =#
#= axP[2, 2].ticklabel_format(style="sci", scilimits=(0, 0)) =#
#= axP[2, 3].ticklabel_format(style="sci", scilimits=(0, 0)) =#
#= axP[1, 1].legend() =#
#= for ax in axP =#
#=     ax.set_xlim([-1e-13, 1e-13]) =#
#= end =#
#= figP.tight_layout() =#
#= figP.savefig("evolProfiles.png") =#
