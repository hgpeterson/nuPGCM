using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function run(; bl = false)
    # parameters
    f = -5.5e-5
    L = 2e5
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
    
    # topography: bowl
    no_net_transport = true
    H0 = 5.5e3
    A = 3e3
    l = 50e3
    H_func(x) = H0 - A*exp(-x^2/2/l^2)
    Hx_func(x) = A*x/l^2*exp(-x^2/2/l^2)

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
    Δt = 1*secs_in_day
    t_plot = 1*secs_in_year
    t_save = 20*secs_in_year
    
    # create model struct
    m = ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup(m)

    # set initial state
    b = zeros(nξ, nσ)
    for i=1:nξ
        b[i, :] = cumtrapz(m.N2[i, :], m.z[i, :]) .- trapz(m.N2[i, :], m.z[i, :])
    end
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2D(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, 100*secs_in_year, t_plot, t_save) 

    return m, s
end

m, s = run()
# m, s = run(; bl=true)

################################################################################
# plots
################################################################################

setup_file = string(out_folder, "setup.h5")
m = load_setup_2D(setup_file)
state_files = string.(out_folder, "state", 0:5, ".h5")
iξ = argmin(abs.(m.ξ .- m.L/4))
profile_plot(setup_file, state_files, iξ) 

println("Done.")