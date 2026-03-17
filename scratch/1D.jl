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

μϱ = 1e2
α = 1/4
θ = π/4
ε = 1e-1
no_Px = false
no_Py = false
H = 1
f = 1
nz = 2^8
T = μϱ/ε^2/1e2
Δt = 1e-4*T

# Ω = 2π/86400  # s⁻¹
# a = 6.371e6  # m
# β = 2Ω/a  # m⁻¹ s⁻¹
# L = 2π*a*60/360  # m
# f₀ = β*L  # s⁻¹
# H₀ = 4e3  # m
# κ₀ = 1e-5  # m² s⁻¹
# Kₑ = 1000  # m² s⁻¹
# N₀ = 1e-3  # s⁻¹
# ν₀ = Kₑ*f₀^2/N₀^2  # m² s⁻¹
# ε = sqrt(ν₀/f₀/H₀^2)
# μ = ν₀/κ₀
# ϱ = (N₀*H₀/f₀/L)^2
# t₀ = 1/f₀/ϱ  # s
# μϱ = μ*ϱ
# α = 1/4
# Δt = 100*86400/t₀
# θ = π/4
# no_Px = false
# no_Py = false
# f = 0.5
# H = α
# T = μϱ/ε^2/1e2

params = (μϱ=μϱ, α=α, θ=θ, ε=ε, Δt=Δt, no_Px=no_Px, no_Py=no_Py, H=H, f=f, nz=nz, T=T)

# solve
u, v, Px, Py, b, t, z = OneDModel.solve(params)

# plot
filename = joinpath(@__DIR__, "images/1d.png")
fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
ax[1].set_ylabel(L"Vertical coordinate $z$")
ax[1].set_xlabel("Flow")
ax[2].set_xlabel(L"Stratification $\partial_z b$")
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
ax[2].plot(1 .+ bz, z, "k-")
if t !== nothing
    ax[1].set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(t))))
end
savefig(filename)
@info "Saved '$filename'"
plt.close()

ν = f^2 ./ ( α * (1 .+ differentiate(b, z)) )
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