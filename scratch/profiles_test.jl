using nuPGCM
using JLD2
using PyPlot

include("../plots/derivatives.jl")

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

# dictionary of variables
u = Dict()
v = Dict()
w = Dict()
b = Dict()
z = Dict()

# load 1D profiles
file = jldopen("../out/data/state1D.jld2", "r")
# file = jldopen("../out/data/state1D_U.jld2", "r")
u["1D"] = file["u"]
v["1D"] = file["v"]
w["1D"] = file["w"]
b["1D"] = file["b"]
z["1D"] = file["z"]
close(file)

# load 2D profiles
file = jldopen("../out/data/state2D_column.jld2", "r")
u["2D"] = file["u"]
v["2D"] = file["v"]
w["2D"] = file["w"]
b["2D"] = file["b"]
z["2D"] = file["z"]
close(file)

# # load 3D profiles
# file = jldopen("../out/data/state3D_column.jld2", "r")
# u["3D"] = file["u"]
# v["3D"] = file["v"]
# w["3D"] = file["w"]
# b["3D"] = file["b"]
# z["3D"] = file["z"]
# close(file)

# plot
pc = 1/6
fig, ax = plt.subplots(1, 4, figsize=(33pc, 33pc/4*1.62), sharey=true)
ax[1].set_ylabel(L"z")
ax[1].set_xlabel(L"u")
ax[2].set_xlabel(L"v")
ax[3].set_xlabel(L"w")
ax[4].set_xlabel(L"\partial_z b")
ax[4].set_xlim(0, 1.1)
for a ∈ ax[1:3]
    a.spines["left"].set_visible(false)
    a.axvline(0, color="k", lw=0.5)
end
for a ∈ ax 
    a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
end
# for dim ∈ ["1D", "2D", "3D"]
for dim ∈ ["1D", "2D"]
    ax[1].plot(u[dim], z[dim], label=dim)
    ax[2].plot(v[dim], z[dim], label=dim)
    ax[3].plot(w[dim], z[dim], label=dim)
    ax[4].plot(1 .+ differentiate(b[dim], z[dim]), z[dim], label=dim)
end
ax[1].legend()
savefig("../out/images/profiles.png")
println("../out/images/profiles.png")
plt.close()

