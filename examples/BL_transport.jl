using nuPGCM.Numerics
using nuPGCM.OneDimensionalModel
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function run_single_column(U; bl=false)
    # parameters (see `setup.jl`)
    f = -5.5e-5
    N2 = 1e-6
    nz = 2^8
    H = 2e3
    θ = 2.5e-3                 # ridge
    # θ = atan(sqrt(0.5*f^2/N2))   # S = 0.5
    # θ = atan(sqrt(0.001*f^2/N2)) # S = 0.001
    # H = 3673.32793219601       # seamount
    # θ = -0.03639128788776821   # seamount
    transport_constraint = true
    # transport_constraint = false
    U = [U]

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
    Δt = 10*secs_in_day
    t_save = 3*secs_in_year
    
    # create model struct
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transport_constraint, U)

    # save and log params
    save_setup_1DPG(m)

    # set initial state
    b = zeros(nz)
    χ, u, v = invert(m, b)
    i = [1]
    s = ModelState1DPG(b, χ, u, v, i)

    # solve transient
    evolve!(m, s, 15*secs_in_year, t_save) 
    
    # solve steady state
    # steady_state(m)

    return m, s
end

function get_transports(m::ModelSetup1DPG, s::ModelState1DPG)
    δ, μ, S, q = get_BL_params(m)
    # i_q = argmin(abs.(m.z .- m.z[1] .- q^-1))
    i_q = argmin(abs.(m.z .- m.z[1] .- 10*q^-1))
    U_B = s.χ[i_q]
    U_I = s.χ[end] - U_B
    return U_B, U_I
end

################################################################################
# run
################################################################################

Us = [0, 1e-2, 1e-1, 1]
U_Bs = zeros(size(Us, 1))
U_Is = zeros(size(Us, 1))
for i=1:size(Us, 1)
    m, s = run_single_column(Us[i])
    U_Bs[i], U_Is[i] = get_transports(m, s)
end

################################################################################
# plots
################################################################################

setup_file = string(out_folder, "setup.h5")
# state_files = string.(out_folder, "state", -1:5, ".h5")
state_files = string.(out_folder, "state", 0:5, ".h5")
profile_plot(setup_file, state_files)

fig, ax = subplots(1)
ax.loglog(Us, U_Bs./Us, "o", label=L"$U^\xi_\mathrm{B}$")
ax.loglog(Us, U_Is./Us, "o", label=L"$U^\xi_\mathrm{I}$")
ax.legend()
ax.set_xlabel(L"Transport $U^\xi$ (m$^2$ s$^{-1}$)")
ax.set_ylabel(L"BL and Interior Transports (m$^2$ s$^{-1}$)")
savefig(string(out_folder, "BL_transport.png"))
println(string(out_folder, "BL_transport.png"))
plt.close()

println("Done.")