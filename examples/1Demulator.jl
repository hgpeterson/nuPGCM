using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function single_column_inversion(; bl=false)
    # parameters (see `setup.jl`)
    f = 8.753044701640954e-5 
    N2 = 1e-6
    nz = 2^8
    H = 1613.8975181510887
    θ = 0.0024260899218681575
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

m1D, s1D = single_column_inversion()

# get 3D u, v
ξ₀ = 4e6
η₀ = 0
uξ = zeros(m.nσ)
uη = zeros(m.nσ)
for j=1:m.nσ
    uξ[j] = nuPGCM.fem_evaluate(m, s.uξ[:, j], ξ₀, η₀)
    uη[j] = nuPGCM.fem_evaluate(m, s.uη[:, j], ξ₀, η₀)
end

# plot u
fig, ax = subplots(1, 2, figsize=(2*1.955, 3.176), sharey=true)
ax[1].set_xlabel(L"Zonal velocity $u^\xi$ (10$^{-3}$ m s$^{-1}$)")
ax[2].set_xlabel(L"Meridional velocity $u^\eta$ (10$^{-3}$ m s$^{-1}$)")
ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")
ax[1].plot(1e3*s1D.u, m1D.z/1e3, label="1D")
ax[1].plot(1e3*uξ, m1D.z/1e3, label="3D")
ax[2].plot(1e3*s1D.v, m1D.z/1e3, label="1D")
ax[2].plot(1e3*uη, m1D.z/1e3, label="3D")
ax[1].legend()
savefig("images/uv_column.png")
println("images/uv_column.png")
plt.close()