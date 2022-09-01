using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function emulate_2D(; bl = false)
    # parameters (see `setup.jl`)
    f = 1e-4
    L = 5e6
    nξ = 2^8 
    nσ = 2^8
    coords = "axisymmetric"
    periodic = false

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L)
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: sine
    no_net_transport = true
    H₀ = 2e3
    Δ = L/5 
    G(x) = 1 - exp(-x^2/(2*Δ^2)) 
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    w = 4*Δ
    c = 0
    G_bump(x) = if c - w < x < c + w return exp(1 - w^2/(w^2 - (x - c)^2)) else return 0 end 
    Gx_bump(x) = -2*(x - c)*w^2*G_bump(x)/(w^2 - (x - c)^2)^2
    # H_func(x)  = H₀ + 0*x
    # Hx_func(x) = 0*x
    H_func(x)  = H₀*G(x - L) + 500
    Hx_func(x) = H₀*Gx(x - L)
    # H_func(x)  = H₀ - 2e2*G_bump(x) 
    # Hx_func(x) =    - 2e2*Gx_bump(x)

    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    # κ0 = 1e-1
    # κ1 = 0
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    μ = 1e0
    ν_func(ξ, σ) = μ*κ_func(ξ, σ)

    # stratification
    N2 = 1e-6
    N2_func(ξ, σ) = N2
    
    # timestepping
    Δt = 0.
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # set initial state
    b = zeros(nξ, nσ)
    for j=1:nσ
        b[:, j] .= m.N2[:, j].*m.H*m.σ[j] + 0.1*m.N2[:, j].*m.H*exp(-(m.σ[j] + 1)/0.1)
    end
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    return m, s
end

m2D, s2D = emulate_2D()
save_setup(m2D, "setup2D.h5")
save_state(s2D, "state2D.h5")

ridge_plot(m2D, s2D, s2D.uξ, "", L"Zonal velocity $u^x$ (m s$^{-1}$)"; style="pcolormesh")
savefig("images/ux2D.png")
println("images/ux2D.png")
plt.close()

ridge_plot(m2D, s2D, s2D.uη, "", L"Meridional velocity $u^y$ (m s$^{-1}$)"; style="pcolormesh")
savefig("images/uy2D.png")
println("images/uy2D.png")
plt.close()