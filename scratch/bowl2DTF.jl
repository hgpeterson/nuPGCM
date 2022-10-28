using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function run(; bl = false)
    # parameters
    f = -5.5e-5
    L = 2e6
    nξ = 2^8 + 1 
    nσ = 2^8
    coords = "cartesian"
    periodic = true

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(-L/2:L/nξ:(L/2 - L/nξ))
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography
    no_net_transport = true
    H0 = 2e3
    # H_func(x) = H0*(sqrt(2 - (2*x/L)^2) - 1) + 500
    # Hx_func(x) = -4*x*H0/L^2/sqrt(2 - (2*x/L)^2)
    Δ = L/5
    G(x) = 1 - exp(-x^2/2/Δ^2)
    Gx(x) = x/Δ^2*G(x)
    H_func(x) = H0*G(x - L/2)*G(x + L/2) + H0/16
    Hx_func(x) = H0*Gx(x - L/2)*G(x + L/2) + H0*G(x - L/2)*Gx(x + L/2)
    # H0 = 2e3
    # amp = 0.4*H0
    # H_func(x) = H0 + amp*cos(2*π*x/L)
    # Hx_func(x) = -2*π/L*amp*sin(2*π*x/L)

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
    
    # timestepping
    Δt = 10*secs_in_day
    t_plot = 3*secs_in_year
    t_save = 3*secs_in_year
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup(m)

    # set initial state
    b = zeros(nξ, nσ)
    for i=1:nξ
        b[i, :] = cumtrapz(m.N2[i, :], m.z[i, :]) .- trapz(m.N2[i, :], m.z[i, :])
    end
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, 15*secs_in_year, t_plot, t_save) 

    return m, s
end

m, s = run()

################################################################################
# plots
################################################################################

# setup_file = string(out_folder, "setup.h5")
# m = load_setup_2D(setup_file)
# state_files = string.(out_folder, "state", 0:5, ".h5")
# iξ = argmin(abs.(m.ξ .- 3*m.L/4))
# profile_plot(setup_file, state_files, iξ) 

println("Done.")