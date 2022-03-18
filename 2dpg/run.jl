################################################################################
# Run 2D νPGCM
################################################################################

# plotting stylesheet
using PyPlot
plt.style.use("../plots.mplstyle")
close("all")
pygui(false)

# run setup
include("setup.jl")

function run_ridge(; bl = false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    ξVariation = true
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
    global symmetry = true
    H0 = 2e3
    amp = 0.4*H0
    H_func(x) = H0 + amp*cos(2*π*x/L)
    Hx_func(x) = -2*π/L*amp*sin(2*π*x/L)

    # # topography: skew gaussian
    # global symmetry = false
    # H0 = 2.5e3 
    # amp = 0.65*H0 
    # ϕ(s) = exp(-s^2/2) 
    # Φ(s) = 1/2*(1 + erf(s/√2)) 
    # α = 3 
    # m = 2e6/3 
    # ω = 2e6/5 
    # H_func(x) = H0 - amp*ϕ((x - m)/ω)*Φ(α*(x - m)/ω) 
    # Hx_func(x) = -amp/ω*(α/sqrt(2π)*ϕ(α*(x - m)/ω)*ϕ((x - m)/ω) - (x - m)/ω*ϕ((x - m)/ω)*Φ(α*(x - m)/ω)) 

    # # topography: bump
    # global symmetry = false
    # H0 = 2.5e3 
    # amp = 1.5e3
    # wid = L/4
    # function bump(s)
    #     if abs(s) >= 1
    #         return 0
    #     else
    #         return exp(1 - 1/(1 - s^2)) 
    #     end
    # end
    # ∂bump(s) = -2*s/(1 - s^2)^2*bump(s)
    # skewBump(s) = (s + 1)*bump(s)
    # ∂skewBump(s) = bump(s) + (s + 1)*∂bump(s)
    # H_func(x) = H0 - amp*skewBump((x - L/2)/wid)
    # Hx_func(x) = -amp/wid*∂skewBump((x - L/2)/wid)

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
    t_plot = 3*secs_in_year
    t_save = 3*secs_in_year
    
    # create model struct
    m = ModelSetup2DPG(bl, f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup_2DPG(m)

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

function run_seamount(; bl=false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    ξVariation = true
    L = 2e5
    nξ = 2^8 + 1 
    nσ = 2^8
    coords = "axisymmetric"
    periodic = false

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L) # avoid r = 0
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography
    global symmetry = true
    H0 = 5.5e3
    H1 = 3e3
    L0 = L/4
    H_func(x) = H0 - H1*exp(-x^2/(2*L0^2))
    Hx_func(x) = H1*x/L0^2*exp(-x^2/(2*L0^2))
    
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
    # N2 = 1e-6*exp(H_func(L/4)/δ) # match bottom strat with const N2 at center of seamount flank
    # N2_func(ξ, σ) = N2*exp(H_func(ξ)*σ/δ)
    
    # timestepping
    Δt = 1*secs_in_day
    t_plot = 20*secs_in_year
    t_save = 20*secs_in_year
    
    # create model struct
    m = ModelSetup2DPG(bl, f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup_2DPG(m)

    # set initial state
    b = zeros(nξ, nσ)
    for i=1:nξ
        b[i, :] = cumtrapz(m.N2[i, :], m.z[i, :]) .- trapz(m.N2[i, :], m.z[i, :])
    end
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # debug: what's the max slope Burger number?
    ϱ = @. m.N2[:, 1]/m.f^2*m.Hx^2
    println("ϱₘₐₓ = ", maximum(ϱ))

    # solve
    evolve!(m, s, 100*secs_in_year, t_plot, t_save) 

    return m, s
end

m, s = run_ridge()
# m, s = run_ridge(; bl=true)
# m, s = run_seamount()
# m, s = run_seamount(; bl=true)

################################################################################
# plots
################################################################################

setup_file = string(out_folder, "setup.h5")
m = load_setup_2DPG(setup_file)
state_files = string.(out_folder, "state", 0:5, ".h5")
iξ = argmin(abs.(m.ξ .- m.L/4))
profile_plot(setup_file, state_files, iξ) 

println("Done.")
