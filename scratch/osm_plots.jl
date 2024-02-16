using nuPGCM
using PyPlot
using Printf
using PyCall
lines = pyimport("matplotlib.lines")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

if !isdir("../output")
    mkdir("../output")
end
set_out_folder("../output")
if !isdir("$out_folder/data")
    mkdir("$out_folder/data")
end
if !isdir("$out_folder/images")
    mkdir("$out_folder/images")
end


### load m, s

# m = load_setup_3D("../../group_dir/sim011/adv_on/output/data/setup.h5")
# s = load_state_3D(m, "../../group_dir/sim011/adv_on/output/data/state10.h5")

# m = load_setup_3D("../../group_dir/sim012/adv_on/output/data/setup.h5")
# s = load_state_3D(m, "../../group_dir/sim012/adv_on/output/data/state10.h5")

### f_over_H.png, curl_tau_b.png

# nuPGCM.barotropic_terms_BL(m, s)

### U.png

# Ux, Uy = nuPGCM.compute_U(s.Ψ)
# Ux = FEField(Ux)
# Uy = FEField(Uy)
# x = -1.0:0.01:1.0
# y = 0
# fig, ax = plt.subplots(1)
# ax.axhline(0, c="k", ls="--", lw=0.25)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-0.06, 0.06)
# ax.set_xlabel(L"Zonal coordinate $x$")
# ax.set_ylabel("Transport")
# ax.plot(x, [Ux([x₀, y]) for x₀ ∈ x], "C0-")
# ax.plot(x, Ux_beta, "C0--")
# ax.plot(x, [Uy([x₀, y]) for x₀ ∈ x], "C1-")
# ax.plot(x, Uy_beta, "C1--")
# custom_handles = [lines.Line2D([0], [0], c="C0", ls="-",  lw=1),
#                   lines.Line2D([0], [0], c="C1", ls="-",  lw=1),
#                   lines.Line2D([0], [0], c="k",  ls="-",  lw=1),
#                   lines.Line2D([0], [0], c="k",  ls="--", lw=1)]
# custom_labels = [L"U^x", L"U^y", L"$f$-plane", L"$\beta$-plane"]
# ax.legend(custom_handles, custom_labels, ncol=2)
# savefig("$out_folder/images/U.png")
# println("$out_folder/images/U.png")

### psi.png

# nuPGCM.quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", filename="$out_folder/images/psi.png")

### ux.png, uy.png

# nuPGCM.plot_u(m, s, 0; title="")

### baroclinic_U.png

ε² = 1e-4
f = 1
H = 1
nσ = 2^8
σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
z = σ*H
ν = @. 1e-2 + exp(-H*(σ + 1)/0.1)
p = σ
t = [i + j - 1 for i=1:nσ-1, j=1:2]
e = Dict("bot"=>[1], "sfc"=>[nσ])
g = Grid(Line(order=1), p, t, e)

A = nuPGCM.build_baroclinic_LHS(g, ν, H, ε², f)
r = nuPGCM.build_baroclinic_RHS(g, zeros(2nσ-2), zeros(2nσ-2), 1, 0, 0, 0)
sol = A\r
ωx_Ux = sol[0nσ+1:1nσ]
ωy_Ux = sol[1nσ+1:2nσ]
χx_Ux = sol[2nσ+1:3nσ]
χy_Ux = sol[3nσ+1:4nσ]
ux_Ux = -differentiate(χy_Ux, σ)/H
uy_Ux = +differentiate(χx_Ux, σ)/H

r = nuPGCM.build_baroclinic_RHS(g, zeros(2nσ-2), zeros(2nσ-2), 0, 1, 0, 0)
sol = A\r
ωx_Uy = sol[0nσ+1:1nσ]
ωy_Uy = sol[1nσ+1:2nσ]
χx_Uy = sol[2nσ+1:3nσ]
χy_Uy = sol[3nσ+1:4nσ]
ux_Uy = -differentiate(χy_Uy, σ)/H
uy_Uy = +differentiate(χx_Uy, σ)/H

fig, ax = plt.subplots(1, 3, figsize=(6, 3.2), sharey=true)
# for a ∈ ax
#     a.spines["left"].set_visible(false)
#     a.axvline(0, lw=0.5, c="k")
#     a.set_xticks([0, 1])
#     a.set_yticks([])
#     a.text(-0.1, 0, s=L"z")
# end
# ax[1].set_xticklabels([L"0", L"U^x"])
# ax[2].set_xticklabels([L"0", L"U^y"])
ax[1].plot(χx_Ux,  z,       label=L"\chi^x_{U^x}")
ax[1].plot(χy_Ux,  z,       label=L"\chi^y_{U^x}")
ax[1].plot(-χx_Uy, z, "--", label=L"-\chi^x_{U^y}")
ax[1].plot(χy_Uy,  z, "--", label=L"\chi^y_{U^y}")
ax[1].legend()
ax[2].plot(ux_Ux,  z,       label=L"u^x_{U^x}")
ax[2].plot(uy_Ux,  z,       label=L"u^y_{U^x}")
ax[2].plot(-ux_Uy, z, "--", label=L"-u^x_{U^y}")
ax[2].plot(uy_Uy,  z, "--", label=L"u^y_{U^y}")
ax[2].legend()
ax[3].plot(ωx_Ux,  z,       label=L"\omega^x_{U^x}")
ax[3].plot(ωy_Ux,  z,       label=L"\omega^y_{U^x}")
ax[3].plot(-ωx_Uy, z, "--", label=L"-\omega^x_{U^y}")
ax[3].plot(ωy_Uy,  z, "--", label=L"\omega^y_{U^y}")
ax[3].legend()
ax[1].set_xlabel("Streamfunction")
ax[2].set_xlabel("Velocity")
ax[3].set_xlabel("Vorticity")
ax[1].set_ylabel(L"z")
# ax[1].text(1.05, -0.9, s=L"u^x")
# ax[1].text(0.1, -0.95, s=L"u^y")
# ax[2].text(-0.3, -0.95, s=L"u^x")
# ax[2].text(1.05, -0.9, s=L"u^y")
savefig("$out_folder/images/omega_chi.png")
println("$out_folder/images/omega_chi.png")
plt.close()

# fig, ax = plt.subplots(1, figsize=(2, 3.2))
# ax.spines["left"].set_visible(false)
# ax.axvline(0, lw=0.5, c="k")
# ax.set_xticks([0, 1])
# ax.set_xticklabels([L"0", L"U^x"])
# ax.set_yticks([])
# ax.plot(ux_Ux, z)
# ax.plot(uy_Ux, z)
# ax.text(-0.1, 0, s=L"z")
# ax.text(1.05, -0.9, s=L"u^x")
# ax.text(0.1, -0.95, s=L"u^y")
# savefig("$out_folder/images/baroclinic_Ux.png")
# println("$out_folder/images/baroclinic_Ux.png")
# plt.close()
# fig, ax = plt.subplots(1, figsize=(2, 3.2))
# ax.spines["left"].set_visible(false)
# ax.axvline(0, lw=0.5, c="k")
# ax.set_xticks([0, 1])
# ax.set_xticklabels([L"0", L"U^y"])
# ax.set_yticks([])
# ax.plot(ux_Uy, z)
# ax.plot(uy_Uy, z)
# ax.text(-0.1, 0, s=L"z")
# ax.text(-0.3, -0.95, s=L"u^x")
# ax.text(1.05, -0.9, s=L"u^y")
# savefig("$out_folder/images/baroclinic_Uy.png")
# println("$out_folder/images/baroclinic_Uy.png")
# plt.close()
# fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
# for a ∈ ax
#     a.spines["left"].set_visible(false)
#     a.axvline(0, lw=0.5, c="k")
#     a.set_xticks([0, 1])
#     a.set_yticks([])
#     a.text(-0.1, 0, s=L"z")
# end
# ax[1].set_xticklabels([L"0", L"U^x"])
# ax[2].set_xticklabels([L"0", L"U^y"])
# ax[1].plot(ux_Ux, z)
# ax[1].plot(uy_Ux, z)
# ax[2].plot(ux_Uy, z)
# ax[2].plot(uy_Uy, z)
# ax[1].text(1.05, -0.9, s=L"u^x")
# ax[1].text(0.1, -0.95, s=L"u^y")
# ax[2].text(-0.3, -0.95, s=L"u^x")
# ax[2].text(1.05, -0.9, s=L"u^y")
# savefig("$out_folder/images/baroclinic_U.png")
# println("$out_folder/images/baroclinic_U.png")
# plt.close()