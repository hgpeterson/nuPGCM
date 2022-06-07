using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function run_ridge(; bl = false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    L = 2e6
    nξ = 2^8 + 1 
    nσ = 2^8
    coords = "cartesian"
    periodic = true

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(0:L/nξ:(L - L/nξ))
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: sine
    no_net_transport = true
    # no_net_transport = false
    H0 = 2e3
    amp = 0.4*H0
    H_func(x) = H0 + amp*cos(2*π*x/L)
    Hx_func(x) = -2*π/L*amp*sin(2*π*x/L)

    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    μ = 2e2
    ν_func(ξ, σ) = μ*κ_func(ξ, σ)

    # stratification
    N2 = 1e-6
    N2_func(ξ, σ) = N2
    # δ = 1000 # decay scale (m)
    # N2 = 1e-6*exp(H_func(L/4)/δ) # match bottom strat with const N2 at center of ridge flank
    # N2_func(ξ, σ) = N2*exp(H_func(ξ)*σ/δ)
    
    # timestepping
    # Δt = 10*secs_in_day
    Δt = 1*secs_in_day
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

m, s = run_ridge()
# m, s = run_ridge(; bl=true)

################################################################################
# plots
################################################################################

setup_file = string(out_folder, "setup.h5")
m = load_setup_2D(setup_file)
state_files = string.(out_folder, "state", 0:5, ".h5")
iξ = argmin(abs.(m.ξ .- m.L/4))
profile_plot(setup_file, state_files, iξ) 

println("Done.")