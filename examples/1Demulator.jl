using nuPGCM
using PyPlot
using Printf
using Dierckx

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function emulate_1D(ξ₀, η₀; bl=false)
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
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(z) = κ0 + κ1*exp(-(z + H)/h)
    κ_z_func(z) = -κ1/h*exp(-(z + H)/h)

    # viscosity
    μ = 1e0
    ν_func(z) = μ*κ_func(z)
    
    # timestepping
    Δt = 0.
    
    # create model struct
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transport_constraint, U)

    # set initial state
    b = @. 0.1*N2*H*exp(-(z + H)/(0.1*H))
    χ, u, v = invert(m, b)
    i = [1]
    s = ModelState1DPG(b, χ, u, v, i)

    return m, s
end

# load 2D
m2D = load_setup_2D("../output/setup2D.h5")
s2D = load_state_2D("../output/state2D.h5")

# comparison points
ξ₀s = 0.75e6:0.5e6:4.75e6
for i=1:size(ξ₀s, 1)
    ξ₀ = ξ₀s[i]
    η₀ = 0
    m1D, s1D = emulate_1D(ξ₀, η₀)

    # get 2D ux, uy
    uξ2D = Spline2D(m2D.ξ, m2D.σ, s2D.uξ).(ξ₀, m2D.σ)
    uη2D = Spline2D(m2D.ξ, m2D.σ, s2D.uη).(ξ₀, m2D.σ)

    # get 3D ux, uy
    uξ3D = zeros(m3D.nσ)
    uη3D = zeros(m3D.nσ)
    for j=1:m3D.nσ
        uξ3D[j] = fem_evaluate(m3D, s3D.uξ[:, j], ξ₀, η₀)
        uη3D[j] = fem_evaluate(m3D, s3D.uη[:, j], ξ₀, η₀)
    end

    # get H
    H = fem_evaluate(m3D, m3D.H, ξ₀, η₀)

    # plot u
    fig, ax = subplots(1, 2, figsize=(2*1.955, 3.176))
    ax[1].set_title(latexstring(L"Comparison point: $x = $", @sprintf("%d", ξ₀/1e3), " km")) 
    ax[1].set_xlabel(L"Zonal velocity $u^x$ ($\times$ 10$^{-3}$ m s$^{-1}$)")
    ax[2].set_xlabel(L"Meridional velocity $u^y$ ($\times$ 10$^{-3}$ m s$^{-1}$)")
    ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")
    ax[1].plot(1e3*s1D.u, m1D.z/1e3,   label="1D")
    ax[1].plot(1e3*uξ2D,  H*m2D.σ/1e3, label="2D")
    ax[1].plot(1e3*uξ3D,  m3D.σ*H/1e3, label="3D", c="k", ls="--", lw=0.5)
    ax[2].plot(1e3*s1D.v, m1D.z/1e3,   label="1D")
    ax[2].plot(1e3*uη2D,  H*m2D.σ/1e3, label="2D")
    ax[2].plot(1e3*uη3D,  m3D.σ*H/1e3, label="3D", c="k", ls="--", lw=0.5)
    ax[1].legend()
    ax[1].set_ylim([-H/1e3, (-H + 100)/1e3])
    savefig("images/ux_uy_column$i.png")
    println("images/ux_uy_column$i.png")
    plt.close()
end