using nuPGCM
using nuPGCM.Numerics
using nuPGCM.OneDModel
using Printf
using PyPlot
using JLD2

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

# parameters

# μϱ = 1
# α = 1/4
# θ = atan(2α)
# ε = 0.1
# N² = 1/α
# no_Px = false
# no_Py = false
# H = α
# f = 1
# nz = 2^8
# T = μϱ/ε^2/1e2
# Δt = 1e-4*T
# eddy_param = false

Ω = 2π/86400  # s⁻¹
a = 6.371e6  # m
β = 2Ω/a  # m⁻¹ s⁻¹
L = 2π*a*60/360  # m
f₀ = β*L  # s⁻¹
H₀ = 4e3  # m
κ₀ = 1e-5  # m² s⁻¹
Kₑ = 1000  # m² s⁻¹
N₀ = 1e-3  # s⁻¹
ν₀ = Kₑ*f₀^2/N₀^2  # m² s⁻¹
# ε = sqrt(ν₀/f₀/H₀^2)
ε /= 10
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2
t₀ = 1/f₀/ϱ  # s
μϱ = μ*ϱ
α = 1/4
Δt = 100*86400/t₀
θ = atan(2α)
no_Px = false
no_Py = false
f = 0.5
H = α
T = μϱ/ε^2/1e2
eddy_param = false

params = (μϱ=μϱ, α=α, θ=θ, ε=ε, N²=N², Δt=Δt, no_Px=no_Px, no_Py=no_Py, H=H, f=f, nz=nz, T=T)

# solve
u, v, Px, Py, b, t, z = OneDModel.solve(params; eddy_param)

# plot
filename = joinpath(@__DIR__, "images/1d.png")
fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
ax[1].set_ylabel(L"Vertical coordinate $z$")
ax[1].set_xlabel("Flow")
ax[2].set_xlabel(L"Stratification $\alpha \partial_z b$")
for a ∈ ax
    a.set_ylim(-H, 0)
    a.spines["left"].set_visible(false)
    a.axvline(0, color="k", lw=0.5)
    a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
end
ax[2].set_yticks([])
bz = differentiate(b, z)
ax[1].plot(u,       z, "C0-", label=L"$u$")
ax[1].plot(v,       z, "C1-", label=L"$v$")
# ax[1].axvline(-Py/f, c="C0", ls="--", lw=0.5, label=L"$-P_y/f$")
# ax[1].axvline(+Px/f, c="C1", ls="--", lw=0.5, label=L"$P_x/f$")
ax[1].legend()
ax[2].plot(α*(N² .+ bz), z, "k-")
if t !== nothing
    ax[1].set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(t))))
end
savefig(filename)
@info "Saved '$filename'"
plt.close()

ν = abs.(f^2 ./ ( α * (N² .+ differentiate(b, z)) ))
ν[ν .> 1e2 ] .= 1e2
filename = joinpath(@__DIR__, "images/nu.png")
fig, ax = plt.subplots(1, figsize=(2, 3.2))
ax.set_ylabel(L"Vertical coordinate $z$")
ax.set_xlabel(L"Turbulent viscosity $\nu$")
ax.set_ylim(-H, 0)
ax.spines["left"].set_visible(false)
ax.axvline(0, color="k", lw=0.5)
ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
ax.plot(ν, z, "k-")
ax.set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(t))))
savefig(filename)
@info "Saved '$filename'"
plt.close()

filename = joinpath(@__DIR__, "images/slope.png")
x = range(0, 0.5, nz)
xx = repeat(x, 1, nz)
zz = xx*tan(θ) + repeat(z, 1, nz)'
bb = N²*zz + repeat(b, 1, nz)'
uu = repeat(u, 1, nz)'
vmax = maximum(abs.(u))
fig, ax = subplots(1)
img = ax.pcolormesh(xx, zz, uu, cmap="RdBu_r", rasterized=true, shading="auto", vmin=-vmax, vmax=vmax)
cb = colorbar(img, ax=ax, label=L"Cross-slope flow $u$", shrink=0.5)
# cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
levels = range(minimum(bb), maximum(bb), 20)
ax.contour(xx, zz, bb, levels=levels, linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
ax.axis("equal")
ax.spines["left"].set_visible(false)
ax.spines["bottom"].set_visible(false)
ax.set_xticks([0, 0.5])
ax.set_yticks([-α, 0])
savefig(filename)
@info "Saved '$filename'"
plt.close()