using nuPGCM.Numerics
using nuPGCM.ThreeDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# load horizontal mesh
# p, t, e = load_mesh("../meshes/square1.h5")
p, t, e = load_mesh("../meshes/square2.h5")
# p, t, e = load_mesh("../meshes/square3.h5")
# p, t, e = load_mesh("../meshes/circle1.h5")
# p, t, e = load_mesh("../meshes/circle2.h5")
# p, t, e = load_mesh("../meshes/circle3.h5")
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
H(ξ, η) = H₀
Hx(ξ, η) = 0
Hy(ξ, η) = 0
# Δ = Lx/5
# H(ξ, η) = H₀ - H₀*exp(-(abs(ξ) - Lx)^2/(2*Δ^2))
# Hx(ξ, η) = H₀*exp(-(abs(ξ) - Lx)^2/(2*Δ^2))*(abs(ξ) - Lx)/Δ^2*sign(ξ)
# Hy(ξ, η) = 0
# R = Lx
# Δ = Lx/5
# H(ξ, η) = H₀ - H₀*exp(-(sqrt(ξ^2 + η^2) - R)^2/(2*Δ^2))

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
    κ = @. κ0 + κ1*exp(-H(ξ, η)*(σ[i] + 1)/h)
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
for i=1:nσ
    for p₀=p
        ∂b∂x[:, i] += ∂ξ(b, p₀, p, t)
        ∂b∂y[:, i] += ∂η(b, p₀, p, t)
    end
end
for i=1:np
    ∂b∂x[i, :] -= σ.*Hx(ξ[i], η[i])*differentiate(b[i, :], σ)/H(ξ[i], η[i])
    ∂b∂y[i, :] -= σ.*Hy(ξ[i], η[i])*differentiate(b[i, :], σ)/H(ξ[i], η[i])
end

# wind stress 
τ₀ = 0.1 
τξ_wind(ξ, η) = -τ₀*cos(π*η/Ly)
τη_wind(ξ, η) = 0

# barotropic flow
Uξ(ξ, η) = 1e-2
Uη(ξ, η) = 0

# get baroclinic_LHS matrices
baroclinic_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, np) 
for i=1:np 
    baroclinic_LHSs[i] = get_baroclinic_LHS(ν, f₀ + β*η[i], H(ξ[i], η[i]), σ)
end  

# get baroclinic_RHS vectors
baroclinic_RHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, np) 
for i=1:np
    get_baroclinic_RHS(∂b∂x[i, :], ∂b∂y[i, :], τξ_wind(ξ[i], η[i]), τη_wind(ξ[i], η[i]), Uξ(ξ[i], η[i]), Uη(ξ[i], η[i]))
end

# solve system
τξ, τη = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs)

# convert to uξ, uη
uξ = zeros(np, nσ)
uη = zeros(np, nσ)
for i=1:np
    uξ[i, :], uη[i, :] = get_uξ_uη(τξ[i, :], τη[i, :], σ, H(ξ[i], η[i]), ν[i, :])
end
