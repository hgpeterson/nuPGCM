using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# load mesh
# p, t, e = load_mesh("../meshes/square.h5")
p, t, e = load_mesh("../meshes/circle.h5")

# widths of basin
Lx = 5e6
Ly = 5e6

# rescale p
p[:, 1] *= Lx
p[:, 2] *= Ly

# show mesh
ax = tplot(p, t)
savefig("mesh.png")

# constant H
H(ξ, η) = 4e3
Hx(ξ, η) = 0
Hy(ξ, η) = 0

# coriolis parameter f = f₀ + βη
f₀ = 1e-4
β = 1e-11

# no JEBAR term
JEBAR(ξ, η) = 0

# wind stress and its curl
τ_ξ(ξ, η) = 0.1*cos(π*η/Ly)
τ_η(ξ, η) = 0
curl_τ(ξ, η) = -0.1*π/Ly*sin(π*η/Ly)

# right-hand-side forcing
F(ξ, η) = JEBAR(ξ, η) + curl_τ(ξ, η)/H(ξ, η)

# bottom stress from baroclinic solution
τₜ_ξ(ξ, η) = 1e-5
τₜ_η(ξ, η) = 1e-5

# get barotropic_LHS
barotropic_LHS = get_barotropic_LHS(p, t, e, f₀, β, H, Hx, Hy, τₜ_ξ, τₜ_η)

# get barotropic_RHS
barotropic_RHS = get_barotropic_RHS(p, t, e, F)

# solve
Ψ = barotropic_LHS\barotropic_RHS

# plot Ψ
fig, ax, im = tplot(p, t, Ψ)
cb = colorbar(im, ax=ax, label=L"streamfunction $\Psi$ (m$^3$ s$^{-1}$)")
cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
ax.tricontour(p[:, 1], p[:, 2], t .- 1, Ψ, linewidths=0.25, colors="k", linestyles="-")
ax.set_xlabel(L"Horizontal coordintate $\xi$ (m)")
ax.set_ylabel(L"Horizontal coordintate $\eta$ (m)")
ax.axis("equal")
savefig("psi.png")
plt.close()

# plot wind stress
η = -Ly:2*Ly/100:Ly
fig, ax = subplots()
ax.plot(τ_ξ.(0, η), η)
ax.set_xlabel(L"Wind Stress $\tau^\xi$ (N m$^{-2}$)")
ax.set_ylabel(L"Horizontal coordintate $\eta$ (m)")
savefig("tau.png")
plt.close()