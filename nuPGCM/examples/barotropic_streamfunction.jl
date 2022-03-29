using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# load mesh
# p, t, e = load_mesh("../meshes/square1.h5")
# p, t, e = load_mesh("../meshes/square2.h5")
# p, t, e = load_mesh("../meshes/circle1.h5")
p, t, e = load_mesh("../meshes/circle2.h5")

# widths of basin
Lx = 5e6
Ly = 5e6

# rescale p
p[:, 1] *= Lx
p[:, 2] *= Ly

# depth H
H₀ = 4e3
R = Lx
Δ = R/5
# H(ξ, η) = H₀
# Hx(ξ, η) = 0
# Hy(ξ, η) = 0
H(ξ, η) = H₀ - H₀*exp(-(abs(ξ) - Lx)^2/(2*Δ^2))
Hx(ξ, η) = H₀*exp(-(abs(ξ) - Lx)^2/(2*Δ^2))*(abs(ξ) - Lx)/Δ^2*sign(ξ)
Hy(ξ, η) = 0
# H(ξ, η) = H₀ - H₀*exp(-(sqrt(ξ^2 + η^2) - R)^2/(2*Δ^2))

# coriolis parameter f = f₀ + βη
f₀ = 0
β = 1e-11

# JEBAR term
JEBAR(ξ, η) = 0

# wind stress and its curl
τ_ξ(ξ, η) = -0.1*cos(π*η/Ly)
τ_η(ξ, η) = 0
curl_τ(ξ, η) = 0.1*π/Ly*sin(π*η/Ly)

# right-hand-side forcing
F(ξ, η) = JEBAR(ξ, η) + curl_τ(ξ, η)/H(ξ, η)

# bottom stress from baroclinic solution
τₜ_ξ(ξ, η) = 5e-6
τₜ_η(ξ, η) = 5e-6

# get barotropic_LHS
barotropic_LHS = get_barotropic_LHS(p, t, e, f₀, β, H, Hx, Hy, τₜ_ξ, τₜ_η)

# get barotropic_RHS
barotropic_RHS = get_barotropic_RHS(p, t, e, F)

# solve
Ψ = barotropic_LHS\barotropic_RHS

# plot Ψ
plot_horizontal(p, t, Ψ/1e9; clabel=L"Streamfunction $\Psi$ (Sv)")
savefig("psi.png")
println("psi.png")
plt.close()

# plot f/H
f_over_H = @. (f₀ + β*p[:, 2])/(H(p[:, 1], p[:, 2]) + eps())
plot_horizontal(p, t, f_over_H; vext=1e-8, clabel=L"$f/H$ (s m$^{-1}$)")
savefig("f_over_H.png")
println("f_over_H.png")
plt.close()

# plot wind stress
η = -Ly:2*Ly/100:Ly
fig, ax = subplots(figsize=(1.955, 3.167))
ax.axvline(0, c="k", lw=0.5, ls="-")
ax.plot(τ_ξ.(0, η), η/1e3)
ax.set_xlabel(L"Wind stress $\tau^\xi$ (N m$^{-2}$)")
ax.set_ylabel(L"Horizontal coordintate $\eta$ (km)")
ax.spines["left"].set_visible(false)
ax.set_xlim([-0.15, 0.15])
ax.set_xticks(-0.15:0.05:0.15)
savefig("tau.png")
println("tau.png")
plt.close()