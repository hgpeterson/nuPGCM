using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# load mesh
# p, t, e = load_mesh("../meshes/square1.h5")
# p, t, e = load_mesh("../meshes/square2.h5")
# p, t, e = load_mesh("../meshes/square3.h5")
# p, t, e = load_mesh("../meshes/circle1.h5")
# p, t, e = load_mesh("../meshes/circle2.h5")
p, t, e = load_mesh("../meshes/circle3.h5")

# widths of basin
Lx = 5e6
Ly = 5e6

# rescale p
p[:, 1] *= Lx
p[:, 2] *= Ly

# depth H
# H₀ = 4e3
# H(ξ, η) = H₀
# Hx(ξ, η) = 0
# Hy(ξ, η) = 0

# H₀ = 4e3
# Δ = Lx/5
# H(ξ, η) = H₀ - H₀*exp(-(abs(ξ) - Lx)^2/(2*Δ^2))
# Hx(ξ, η) = H₀*exp(-(abs(ξ) - Lx)^2/(2*Δ^2))*(abs(ξ) - Lx)/Δ^2*sign(ξ)
# Hy(ξ, η) = 0

# H₀ = 4e3
# Δ = Lx/5
# G(x) = 1 - exp(-x^2/(2*Δ^2))
# Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
# H(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η)
# Hx(ξ, η) = H₀*Gx(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gx(Lx - ξ)*G(Ly + η)*G(Ly - η)
# Hy(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*Gx(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gx(Ly - η)

H₀ = 4e3
R = Lx
Δ = R/5
G(x) = 1 - exp(-x^2/(2*Δ^2))
Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
H(ξ, η) = H₀*G(sqrt(ξ^2 + η^2) - R)
Hx(ξ, η) = H₀*Gx(sqrt(ξ^2 + η^2) - R)*ξ/sqrt(ξ^2 + η^2)
Hy(ξ, η) = H₀*Gx(sqrt(ξ^2 + η^2) - R)*η/sqrt(ξ^2 + η^2)

# coriolis parameter f = f₀ + βη
f₀ = 0
β = 1e-11

# JEBAR term
JEBAR(ξ, η) = 0

# wind stress and its curl
τ₀ = 0.1 # N m⁻² 
τξ_wind(ξ, η) = -τ₀*cos(π*η/Ly)
τη_wind(ξ, η) = 0
# ∂ξ(τη/H) - ∂η(τξ/H)
curl_τ_wind(ξ, η) = -τ₀*π/Ly*sin(π*η/Ly)/H(ξ, η) - τξ_wind(ξ, η)*Hy(ξ, η)/H(ξ, η)^2  

# right-hand-side forcing
F(ξ, η) = JEBAR(ξ, η) + curl_τ_wind(ξ, η)

# bottom stress from baroclinic solution
r = 5e-6
τξ_tξ_bot(ξ, η) = r
τη_tη_bot(ξ, η) = r
τξ_tη_bot(ξ, η) = r/1e2
τη_tξ_bot(ξ, η) = r/1e2

# get barotropic_LHS
barotropic_LHS = get_barotropic_LHS(p, t, e, f₀, β, H, Hx, Hy, τξ_tξ_bot, τη_tη_bot, τξ_tη_bot, τη_tξ_bot)

# get barotropic_RHS
barotropic_RHS = get_barotropic_RHS(p, t, e, F)

# solve
Ψ = barotropic_LHS\barotropic_RHS

# evaluate Ψ at a point
# println(Numerics.evaluate(Ψ, [0, 2e6], p, t))
# square1: 1.591767429539354e10
# square2: 1.5934110876112041e10
# square3: 1.5945647530011332e10

# # plot flat bottom analytical solution
# ξ = p[:, 1]
# η = p[:, 2]
# Ψ_analytical = @. 1/β*(ξ - Lx + 2Lx*exp(-β*(ξ + Lx)/r))*curl_τ_wind(ξ, η)
# plot_horizontal(p, t, Ψ_analytical/1e9; clabel=L"Streamfunction $\Psi$ (Sv)")
# savefig("psi_analytical.png")
# println("psi_analytical.png")
# plt.close()

# plot Ψ
plot_horizontal(p, t, Ψ/1e9; clabel=L"Streamfunction $\Psi$ (Sv)")
savefig("psi.png")
println("psi.png")
plt.close()

# plot H
plot_horizontal(p, t, H.(p[:, 1], p[:, 2]); clabel=L"$H$ (m)")
savefig("H.png")
println("H.png")
plt.close()

# plot Hx
plot_horizontal(p, t, Hx.(p[:, 1], p[:, 2]); clabel=L"$\partial_x H$ (-)")
savefig("Hx.png")
println("Hx.png")
plt.close()

# plot Hy
plot_horizontal(p, t, Hy.(p[:, 1], p[:, 2]); clabel=L"$\partial_y H$ (-)")
savefig("Hy.png")
println("Hy.png")
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
ax.plot(τξ_wind.(0, η), η/1e3)
ax.set_xlabel(L"Wind stress $\tau^\xi$ (N m$^{-2}$)")
ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
ax.spines["left"].set_visible(false)
ax.set_xlim([-0.15, 0.15])
ax.set_xticks(-0.15:0.05:0.15)
savefig("tau.png")
println("tau.png")
plt.close()
