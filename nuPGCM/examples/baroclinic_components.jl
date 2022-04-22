using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using SuiteSparse
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# load horizontal mesh
p, t, e = load_mesh("../meshes/square1.h5")
np = size(p, 1)

# widths of basin
Lx = 5e6
Ly = 5e6

# rescale p
p[:, 1] *= Lx
p[:, 2] *= Ly
ξ = p[:, 1]
η = p[:, 2]

# linear basis
C₀ = get_linear_basis_coeffs(p, t)

# vertical coordinate
nσ = 2^8
σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  

# depth H
H₀ = 4e3
Δ = Lx/5
G(x) = 1 - exp(-x^2/(2*Δ^2))
Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
H_func(ξ, η) = eps() + H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η)
Hx_func(ξ, η) = H₀*Gx(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gx(Lx - ξ)*G(Ly + η)*G(Ly - η)
Hy_func(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*Gx(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gx(Ly - η)

# coriolis parameter f = f₀ + βη
f₀ = 0
β = 1e-11
f_func(ξ, η) = f₀ + β*η
fy_func(ξ, η) = β

# diffusivity and viscosity
# κ0 = 6e-5
# κ1 = 2e-3
# h = 200
# μ = 1e0
# κ = zeros(np, nσ)
# for i=1:nσ
#     κ[:, i] = @. κ0 + κ1*exp(-H_func.(ξ, η)*(σ[i] + 1)/h)
# end
# ν = μ*κ
ν = 1e1*ones(np, nσ)
κ = 1e1*ones(np, nσ)

# stratification
N² = 1e-6*ones(np, nσ)

# model setup struct
m = ModelSetup3DPG(false, f_func, fy_func, p, t, e, σ, H_func, Hx_func, Hy_func, ν, κ, N², 0.)
println("done")

# # buoyancy field
# b = zeros(np, nσ)

# # buoyancy gradients
# ∂b∂x = zeros(np, nσ)
# ∂b∂y = zeros(np, nσ)
# for i=1:nσ
#     println("i = $i / $nσ")
#     for j=1:np
#         ∂b∂x[:, i] .+= ∂ξ(b, p[j, :], p, t, C₀)
#         ∂b∂y[:, i] .+= ∂η(b, p[j, :], p, t, C₀)
#     end
# end
# for i=1:np
#     println("i = $i / $np")
#     ∂b∂x[i, :] .-= σ*Hx[i].*differentiate(b[i, :], σ)/H[i]
#     ∂b∂y[i, :] .-= σ*Hy[i].*differentiate(b[i, :], σ)/H[i]
# end

# JEBAR term
JEBAR(ξ, η) = 0

# wind stress and its curl
τ₀ = 0.1 # N m⁻² 
τξ_wind(ξ, η) = -τ₀*cos(π*η/Ly)
τη_wind(ξ, η) = 0
# ∂ξ(τη/H) - ∂η(τξ/H)
curl_τ_wind(ξ, η) = -τ₀*π/Ly*sin(π*η/Ly)/H_func(ξ, η) - τξ_wind(ξ, η)*Hy_func(ξ, η)/H_func(ξ, η)^2  

# right-hand-side forcing
F(ξ, η) = JEBAR(ξ, η) + curl_τ_wind(ξ, η)

# get barotropic_RHS
barotropic_RHS = get_barotropic_RHS(p, t, e, F)

# solve
Ψ = m.barotropic_LHS\barotropic_RHS

# plot Ψ
plot_horizontal(p, t, Ψ/1e9; clabel=L"Streamfunction $\Psi$ (Sv)")
savefig("psi.png")
println("psi.png")
plt.close()

# plot H
plot_horizontal(p, t, H_func.(ξ, η); clabel=L"$H$ (m)")
savefig("H.png")
println("H.png")
plt.close()

# plot Hx
plot_horizontal(p, t, Hx_func.(ξ, η); clabel=L"$\partial_x H$ (-)")
savefig("Hx.png")
println("Hx.png")
plt.close()

# plot Hy
plot_horizontal(p, t, Hy_func.(ξ, η); clabel=L"$\partial_y H$ (-)")
savefig("Hy.png")
println("Hy.png")
plt.close()

# plot f/H
f_over_H = @. f_func(ξ, η)/(H_func(ξ, η) + eps())
plot_horizontal(p, t, f_over_H; vext=1e-8, clabel=L"$f/H$ (s m$^{-1}$)")
savefig("f_over_H.png")
println("f_over_H.png")
plt.close()

# plot wind stress
y = -Ly:2*Ly/100:Ly
fig, ax = subplots(figsize=(1.955, 3.167))
ax.axvline(0, c="k", lw=0.5, ls="-")
ax.plot(τξ_wind.(0, y), y/1e3)
ax.set_xlabel(L"Wind stress $\tau^\xi$ (N m$^{-2}$)")
ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
ax.spines["left"].set_visible(false)
ax.set_xlim([-0.15, 0.15])
ax.set_xticks(-0.15:0.05:0.15)
savefig("tau.png")
println("tau.png")
plt.close()
