using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using SuiteSparse
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# # test inversion at single column
# function get_τξ_τη_SC(nσ, ν, f, H, σ)
#     baroclinic_LHS = get_baroclinic_LHS(ν, f, H, σ)
#     baroclinic_RHS = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1.0, 1.0)
#     sol = baroclinic_LHS\baroclinic_RHS
#     imap = reshape(1:2*nσ, (2, nσ)) 
#     τξ = sol[imap[1, :]]
#     τη = sol[imap[2, :]]
#     return τξ, τη
# end

# fig, ax = subplots()
# ax.set_xlabel(L"$\tau$ (m$^2$ s$^{-2}$)")
# ax.set_ylabel(L"$\sigma$")
# nσ = 2^8
# σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
# ν = 2e-3*ones(nσ)
# f = -5.5e-5
# for H=[4.1e3, 3.1e3, 2.1e3, 1.1e3, 0.1e3, 0.01e3]
#     τξ, τη = get_τξ_τη_SC(nσ, ν, f, H, σ)
#     println("H = $H; bottom stress: ($(τξ[1]), $(τη[1]))")
#     ax.plot(abs.(τξ), σ, c="tab:blue")
#     ax.plot(abs.(τη), σ, c="tab:orange", ls="--")
# end
# savefig("debug.png")

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

# vertical coordinate
nσ = 2^8
σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  

# depth H
H₀ = 4e3
Δ = Lx/5
G(x) = 1 - exp(-x^2/(2*Δ^2))
Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
H(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) + 20
Hx(ξ, η) = H₀*Gx(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*Gx(Lx - ξ)*G(Ly + η)*G(Ly - η)
Hy(ξ, η) = H₀*G(Lx + ξ)*G(Lx - ξ)*Gx(Ly + η)*G(Ly - η) - H₀*G(Lx + ξ)*G(Lx - ξ)*G(Ly + η)*Gx(Ly - η)

# coriolis parameter f = f₀ + βη
f₀ = 0
β = 1e-11

# diffusivity and viscosity
κ0 = 6e-5
κ1 = 2e-3
h = 200
μ = 1e0
κ = zeros(np, nσ)
for i=1:nσ
    κ[:, i] = @. κ0 + κ1*exp(-H(ξ, η)*(σ[i] + 1)/h)
end
ν = μ*κ

# buoyancy field
N² = 1e-6
b = zeros(np, nσ)
for i=1:nσ
    b[:, i] = N²*σ[i]*H.(ξ, η)
end

# buoyancy gradients
∂b∂x = zeros(np, nσ)
∂b∂y = zeros(np, nσ)
# for i=1:nσ
#     println("i = $i / $nσ")
#     for j=1:np
#         ∂b∂x[:, i] .+= ∂ξ(b, p[j, :], p, t)
#         ∂b∂y[:, i] .+= ∂η(b, p[j, :], p, t)
#     end
# end
# for i=1:np
#     println("i = $i / $np")
#     ∂b∂x[i, :] .-= σ*Hx(ξ[i], η[i]).*differentiate(b[i, :], σ)/H(ξ[i], η[i])
#     ∂b∂y[i, :] .-= σ*Hy(ξ[i], η[i]).*differentiate(b[i, :], σ)/H(ξ[i], η[i])
# end

# wind stress 
τ₀ = 0.1 
τξ_wind(ξ, η) = -τ₀*cos(π*η/Ly)
τη_wind(ξ, η) = 0

# barotropic flow
Uξ(ξ, η) = 0
Uη(ξ, η) = 0

# get baroclinic_LHS matrices
baroclinic_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, np) 
for i=1:np 
    baroclinic_LHSs[i] = get_baroclinic_LHS(ν[i, :], f₀ + β*η[i], H(ξ[i], η[i]), σ)
end  

# get baroclinic_RHS vectors
baroclinic_RHSs = zeros(np, 2*nσ)
for i=1:np
    baroclinic_RHSs[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 1)
end

# solve system
τξ, τη = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs)

# # convert to uξ, uη
# uξ = zeros(np, nσ)
# uη = zeros(np, nσ)
# for i=1:np
#     uξ[i, :], uη[i, :] = get_uξ_uη(τξ[i, :], τη[i, :], σ, H(ξ[i], η[i]), ν[i, :])
# end

# plot
plot_horizontal(p, t, τξ[:, 1])
savefig("debug.png")
plt.close()
