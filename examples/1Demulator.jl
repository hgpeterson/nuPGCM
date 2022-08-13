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
    f = 1e-4
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
    χ = zeros(nz)
    χ, u, v = invert(m, b, χ)
    i = [1]
    s = ModelState1DPG(b, χ, u, v, i)

    return m, s
end

# load 2D
m2D = load_setup_2D("../output/setup2D.h5")
s2D = load_state_2D("../output/state2D.h5")

# comparison points
iξs = [10, 64, 128, 192, 250]
for i=iξs
    ξ₀ = m2D.ξ[i]
    η₀ = 0
    m1D, s1D = emulate_1D(ξ₀, η₀)

    # get 2D u
    uξ2D = s2D.uξ[i, :]
    uη2D = s2D.uη[i, :]
    uσ2D = s2D.uσ[i, :]

    # get 3D u
    uξ3D = zeros(m3D.nσ)
    uη3D = zeros(m3D.nσ)
    uσ3D = zeros(m3D.nσ)
    for j=1:m3D.nσ
        uξ3D[j] = fem_evaluate(m3D, s3D.uξ[:, j], ξ₀, η₀)
        uη3D[j] = fem_evaluate(m3D, s3D.uη[:, j], ξ₀, η₀)
        uσ3D[j] = fem_evaluate(m3D, s3D.uσ[:, j], ξ₀, η₀)
    end

    # get H
    H = fem_evaluate(m3D, m3D.H, ξ₀, η₀)

    # plot u
    fig, ax = subplots(1, 3, figsize=(3*1.955, 3.176))

    ax[2].set_title(latexstring(L"$x = $", @sprintf("%d", ξ₀/1e3), " km")) 
    ax[1].set_xlabel(latexstring("Zonal velocity\n", L"$u^x$ (m s$^{-1}$)"))
    ax[2].set_xlabel(latexstring("Meridional velocity\n", L"$u^y$ (m s$^{-1}$)"))
    ax[3].set_xlabel(latexstring("Vertical velocity\n", L"$u^\sigma$ (s$^{-1}$)"))
    ax[1].set_ylabel(L"Vertical coordinate $z$ (m)")

    for a in ax
        a.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    end

    ax[1].plot(s1D.u,   m1D.z, label="1D")
    ax[1].plot(uξ2D,  m2D.σ*H, label="2D")
    ax[1].plot(uξ3D,  m3D.σ*H, label="3D", c="k", ls="--", lw=0.5)

    ax[2].plot(s1D.v,   m1D.z, label="1D")
    ax[2].plot(uη2D,  m2D.σ*H, label="2D")
    ax[2].plot(uη3D,  m3D.σ*H, label="3D", c="k", ls="--", lw=0.5)

    # ax[3].plot(s1D.u*tan(m1D.θ),   m1D.z, label="1D")
    ax[3].plot(uσ2D,             H*m2D.σ, label="2D", c="tab:orange")
    ax[3].plot(uσ3D,             H*m3D.σ, label="3D", c="k", ls="--", lw=0.5)

    ax[1].set_ylim([-H, minimum([(-H + 100), 0])])
    # ax[3].set_ylim([-H, minimum([(-H + 100), 0])])
    ax[2].legend()

    plt.subplots_adjust(wspace=0.3)
    savefig("images/u_column$i.png")
    println("images/u_column$i.png")
    plt.close()
end