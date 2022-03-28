using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# load mesh
p, t, e = load_mesh("../meshes/square.h5")
# p, t, e = load_mesh("../meshes/circle.h5")

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
β = 2e-11

# JEBAR term
JEBAR(ξ, η) = 0

# wind stress and its curl
τ_ξ(ξ, η) = 0.1*cos(π*η/Ly)
τ_η(ξ, η) = 0
curl_τ(ξ, η) = -0.1*π/Ly*sin(π*η/Ly)

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
# plot_Ψ(p, t, Ψ; vext=15)
plot_Ψ(p, t, Ψ)

# plot f/H
f_over_H = zeros(size(p, 1))
for i=1:size(p, 1)
    if H(p[i, 1], p[i, 2]) <= 1e3
        f_over_H[i] = 0
    else
        f_over_H[i] = (f₀ + β*p[i, 2])/H(p[i, 1], p[i, 2])
    end
end
fig, ax, im = tplot(p/1e3, t/1e3, f_over_H)
cb = colorbar(im, ax=ax, label=L"$f/H$ (s m$^{-1}$)")
# f_over_H_max = maximum(abs.(f_over_H))
# n = 6
# levels = f_over_H_max*[collect(-(n-1)/n:1/n:-1/n)' collect(1/n:1/n:(n-1)/n)']
# ax.tricontour(p[:, 1]/1e3, p[:, 2]/1e3, t .- 1, f_over_H, linewidths=0.25, colors="k", linestyles="-", levels=levels)
ax.set_xlabel(L"Horizontal coordintate $\xi$ (km)")
ax.set_ylabel(L"Horizontal coordintate $\eta$ (km)")
ax.axis("equal")
savefig("f_over_H.png")
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
plt.close()