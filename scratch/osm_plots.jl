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

# f = 1
# m = load_setup_3D("../../group_dir/sim011/adv_on/output/data/setup.h5")
# s = load_state_3D(m, "../../group_dir/sim011/adv_on/output/data/state10.h5")

# f = 1 + 0.5y
# m = load_setup_3D("../../group_dir/sim014/output/data/setup.h5")
# s = load_state_3D(m, "../../group_dir/sim014/output/data/state10.h5")

# f = 1 + y
# m = load_setup_3D("../../group_dir/sim015/output/data/setup.h5")
# s = load_state_3D(m, "../../group_dir/sim015/output/data/state10.h5")

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

# nuPGCM.quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", filename="$out_folder/images/psi.png", vmax=2.8e-2)
# nuPGCM.quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", filename="$out_folder/images/psi.png")

### ux.png, uy.png

nuPGCM.plot_u(m, s, 0; title="")

### baroclinic_U.png

ε² = 1e-4
f = 1
H = 1
nσ = 2^8
σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
z = σ*H
z_dg = zeros(2nσ-2)
for i ∈ 1:nσ-1
    z_dg[2i-1] = z[i]
    z_dg[2i] = z[i+1]
end
ν = @. 1e-2 + exp(-H*(σ + 1)/0.1)
p = σ
t = [i + j - 1 for i=1:nσ-1, j=1:2]
e = Dict("bot"=>[1], "sfc"=>[nσ])
g = Grid(Line(order=1), p, t, e)

# A = nuPGCM.build_baroclinic_LHS(g, ν, H, ε², f)

# r = nuPGCM.build_baroclinic_RHS(g, zeros(2nσ-2), zeros(2nσ-2), 1, 0, 0, 0)
# sol = A\r
# ωx_Ux = sol[0nσ+1:1nσ]
# ωy_Ux = sol[1nσ+1:2nσ]
# χx_Ux = sol[2nσ+1:3nσ]
# χy_Ux = sol[3nσ+1:4nσ]
# ux_Ux = -differentiate(χy_Ux, σ)/H
# uy_Ux = +differentiate(χx_Ux, σ)/H

# r = nuPGCM.build_baroclinic_RHS(g, zeros(2nσ-2), zeros(2nσ-2), 0, 1, 0, 0)
# sol = A\r
# ωx_Uy = sol[0nσ+1:1nσ]
# ωy_Uy = sol[1nσ+1:2nσ]
# χx_Uy = sol[2nσ+1:3nσ]
# χy_Uy = sol[3nσ+1:4nσ]
# ux_Uy = -differentiate(χy_Uy, σ)/H
# uy_Uy = +differentiate(χx_Uy, σ)/H

# fig, ax = plt.subplots(1, figsize=(2, 3.2))
# ax.spines["left"].set_visible(false)
# ax.axvline(0, lw=0.5, c="k")
# ax.set_xticks([0, 1])
# ax.set_xticklabels([L"0", L"U^x/H"])
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
# ax.set_xticklabels([L"0", L"U^y/H"])
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
# ax[1].set_xticklabels([L"0", L"U^x/H"])
# ax[2].set_xticklabels([L"0", L"U^y/H"])
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

f = 5.5e-5
H = 2e3
ν = @. 6e-5 + 2e-3*exp(-H*(σ + 1)/200)
ε² = ν[1]/f
nσ = 2^8
σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
z = σ*H
p = σ
t = [i + j - 1 for i=1:nσ-1, j=1:2]
e = Dict("bot"=>[1], "sfc"=>[nσ])
g = Grid(Line(order=1), p, t, e)
bx_dg = zeros(2nσ-2)
for i ∈ 1:nσ-1
    bx_dg[2i-1] = bx[i]
    bx_dg[2i] = bx[i+1]
end
A = nuPGCM.build_baroclinic_LHS(g, ν, H, ε², f)

r = nuPGCM.build_baroclinic_RHS(g, bx_dg, zeros(2nσ-2), 0, 0, 0, 0)
sol = A\r
ωx_b = sol[0nσ+1:1nσ]
ωy_b = sol[1nσ+1:2nσ]
χx_b = sol[2nσ+1:3nσ]
χy_b = sol[3nσ+1:4nσ]
ux_b = -differentiate(χy_b, σ)/H
uy_b = +differentiate(χx_b, σ)/H

# three steps
u_TW = zeros(nσ)
v_TW = cumtrapz(bx/f, z)
V = trapz(v_TW, z)
v_TW_no_V = @. v_TW - V/H

# plot
α = 0.4
ax = plotsetup()
ax.plot(u_TW, z, "C0", label=L"u^x")
ax.plot(v_TW, z, "C1", label=L"u^y")
ax.legend(loc="upper left")
savefig("$out_folder/images/baroclinic_b1.png")
println("$out_folder/images/baroclinic_b1.png")
plt.close()
ax = plotsetup()
ax.plot(u_TW, z, "C0", label=L"u^x")
ax.plot(v_TW, z, "C1", alpha=α)
ax.plot(v_TW_no_V, z, "C1", label=L"u^y")
ax.legend(loc="upper left")
savefig("$out_folder/images/baroclinic_b2.png")
println("$out_folder/images/baroclinic_b2.png")
plt.close()
ax = plotsetup()
ax.plot(u_TW, z, "C0", alpha=α)
ax.plot(ux_b, z, "C0", label=L"u^x")
ax.plot(v_TW, z, "C1", alpha=α)
ax.plot(v_TW_no_V, z, "C1", alpha=α)
ax.plot(uy_b, z, "C1", label=L"u^y")
ax.legend(loc="upper left")
savefig("$out_folder/images/baroclinic_b3.png")
println("$out_folder/images/baroclinic_b3.png")
plt.close()
ax = plotsetup()
ax.plot(ux_b, z, "C0")
ax.plot(uy_b, z, "C1")
ax.set_xlim(-1.5e-2, 7e-3)
ax.text(3e-3, -1.9e3, s=L"u^x")
ax.text(-1e-2, -1.75e3, s=L"u^y")
savefig("$out_folder/images/baroclinic_b.png")
println("$out_folder/images/baroclinic_b.png")
plt.close()