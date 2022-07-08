using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function emulate_1D(; bl=false)
    # parameters (see `setup.jl`)
    f = 8.753044701640954e-5 
    N2 = 1e-6
    nz = 2^8
    H = fem_evaluate(m3D, m3D.H, ξ₀, η₀)
    θ = -atan(fem_evaluate(m3D, m3D.Hx, ξ₀, η₀))
    transport_constraint = true
    U = [0.0]

    # grid: chebyshev unless bl
    if bl
        z = collect(-H:H/(nz-1):0) # uniform
    else
        z = @. -H*(cos(pi*(0:nz-1)/(nz-1)) + 1)/2 # chebyshev 
    end
    
    # diffusivity
    κ = 1e-1*ones(nz)
    κ_z = zeros(nz)

    # viscosity
    ν = 1e-1*ones(nz)
    
    # timestepping
    Δt = 1*secs_in_day
    t_save = 3*secs_in_year
    
    # create model struct
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transport_constraint, U)

    # set initial state
    b = @. 0.1*N2*H*exp(-(z + H)/(0.1*H))
    χ, u, v = invert(m, b)
    i = [1]
    s = ModelState1DPG(b, χ, u, v, i)

    return m, s
end

# comparison point
# ξ₀ = 4e6
# ξ₀ = m2D.ξ[206]
ξ₀ = m2D.ξ[50]
η₀ = 0
m1D, s1D = emulate_1D()

# get 2D ux, uy
iξ = argmin(abs.(m2D.ξ .- ξ₀))
uξ2D = s2D.uξ[iξ, :]
uη2D = s2D.uη[iξ, :]

# get 3D ux, uy
uξ3D = zeros(m3D.nσ)
uη3D = zeros(m3D.nσ)
for j=1:m3D.nσ
    uξ3D[j] = fem_evaluate(m3D, s3D.uξ[:, j], ξ₀, η₀)
    uη3D[j] = fem_evaluate(m3D, s3D.uη[:, j], ξ₀, η₀)
end

# plot u
fig, ax = subplots(1, 2, figsize=(2*1.955, 3.176), sharey=true)
ax[1].set_title(latexstring(L"Comparison point: $x = $", @sprintf("%d", ξ₀/1e3), " km")) 
ax[1].set_xlabel(L"Zonal velocity $u^x$ ($\times$ 10$^{-3}$ m s$^{-1}$)")
ax[2].set_xlabel(L"Meridional velocity $u^y$ ($\times$ 10$^{-3}$ m s$^{-1}$)")
ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")
ax[1].plot(1e3*s1D.u, m1D.z/1e3, label="1D")
ax[1].plot(1e3*uξ2D, m1D.z/1e3, label="2D")
ax[1].plot(1e3*uξ3D, m1D.z/1e3, label="3D", c="k", ls="--", lw=0.5)
ax[2].plot(1e3*s1D.v, m1D.z/1e3, label="1D")
ax[2].plot(1e3*uη2D, m1D.z/1e3, label="2D")
ax[2].plot(1e3*uη3D, m1D.z/1e3, label="3D", c="k", ls="--", lw=0.5)
ax[1].legend()
savefig("images/ux_uy_column.png")
println("images/ux_uy_column.png")
plt.close()