using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function emulate_2D(; bl = false)
    # parameters (see `setup.jl`)
    f = 8.753044701640954e-5 
    L = 5e6
    nξ = 2^8 + 1 
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
    # no_net_transport = false
    # H₀ = 4e3
    H₀ = 2e3
    Δ = L/5 # width of gaussian for bathtub
    G(x) = 1 - exp(-x^2/(2*Δ^2)) # gaussian for bathtub
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    H_func(x) = H₀*G(L + x)*G(L - x) + 40
    Hx_func(x) = H₀*Gx(L + x)*G(L - x) - H₀*G(L + x)*Gx(L - x)

    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    μ = 1e0
    ν_func(ξ, σ) = μ*κ_func(ξ, σ)

    # stratification
    N2 = 1e-6
    N2_func(ξ, σ) = N2
    # δ = 1000 # decay scale (m)
    # N2 = 1e-6*exp(H_func(L/4)/δ) # match bottom strat with const N2 at center of ridge flank
    # N2_func(ξ, σ) = N2*exp(H_func(ξ)*σ/δ)
    
    # timestepping
    Δt = 10*secs_in_day
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    # save_setup(m)

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

ridge_plot(m2D, s2D, s2D.uξ, "", L"Zonal velocity $u^x$ (m s$^{-1}$)"; style="pcolormesh")
savefig("images/ux2D.png")
println("images/ux2D.png")
plt.close()

ridge_plot(m2D, s2D, s2D.uη, "", L"Meridional velocity $u^y$ (m s$^{-1}$)"; style="pcolormesh")
savefig("images/uy2D.png")
println("images/uy2D.png")
plt.close()