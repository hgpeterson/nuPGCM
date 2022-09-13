using nuPGCM
using PyPlot
using PyCall

gridspec = pyimport("matplotlib.gridspec")
slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("data/")

function run_ridge(; bl = false)
    # parameters
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
    μ = 1e0
    ν_func(ξ, σ) = μ*κ_func(ξ, σ)

    # stratification
    N2 = 1e-6
    N2_func(ξ, σ) = N2
    
    # timestepping
    Δt = 3.6*secs_in_day
    t_plot = 3*secs_in_year
    t_save = 0.01*secs_in_year
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # save and log params
    save_setup(m)

    # set initial state
    b = zeros(nξ, nσ)
    for i=1:nξ
        b[i, :] = cumtrapz(m.N2[i, :], m.z[i, :]) .- trapz(m.N2[i, :], m.z[i, :])
    end
    χ = zeros(nξ, nσ)
    uξ = zeros(nξ, nσ)
    uη = zeros(nξ, nσ)
    uσ = zeros(nξ, nσ)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, 6*secs_in_year, t_plot, t_save) 

    return m, s
end

function plot_anim(setup_file, state_files)
    m = load_setup_2D(setup_file)
    ix = argmin(abs.(m.ξ .- m.L/4))
    for i in eachindex(state_files)
        # load 
        s = load_state_2D(state_files[i])

        # figure
        fig = figure(figsize=(25*pc, 25*pc))
        gs = gridspec.GridSpec(2, 6)
        ax1 = fig.add_subplot(get(gs, (0, slice(0, 3))))
        ax2 = fig.add_subplot(get(gs, (0, slice(3, 6))))
        ax3 = fig.add_subplot(get(gs, (1, slice(0, 2))))
        ax4 = fig.add_subplot(get(gs, (1, slice(2, 4))))
        ax5 = fig.add_subplot(get(gs, (1, slice(4, 6))))
        fig.suptitle(latexstring(L"$t = $", @sprintf("%.1f years", (s.i[1] - 1)*m.Δt/secs_in_year)))

        # ridge plots
        ridge_plot(m, s, 1e3*s.χ,  "", L"Streamfunction $\chi$ ($\times 10^{-3}$ m$^2$ s$^{-1}$)"; ax=ax1, vext=2.0, cb_orientation="horizontal", pad=0.25)
        ridge_plot(m, s, 1e2*s.uη, "", L"Along-ridge flow $u^y$ ($\times 10^{-2}$ m s$^{-1}$)"; ax=ax2, style="pcolormesh", vext=2, cb_orientation="horizontal", pad=0.25)
        ax1.plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
        ax2.plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
        ax2.set_ylabel("")
        ax2.set_yticklabels([])

        # profile plots
        ax3.set_ylabel(L"Vertical coordinate $z$ (km)")
        ax4.set_yticklabels([])
        ax5.set_yticklabels([])
        ax3.set_xlabel(string(L"Streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
        ax4.set_xlabel(string(L"Along-ridge flow $u^y$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
        ax5.set_xlabel(string(L"Stratification $\partial_z b$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))
        bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax3.plot(1e3*s.χ[ix, :],  m.z[ix, :]/1e3)
        ax4.plot(1e2*s.uη[ix, :], m.z[ix, :]/1e3)
        ax5.plot(1e6*bz,          m.z[ix, :]/1e3)

        # limits
        ax3.set_ylim(-2, 0)
        ax4.set_ylim(-2, 0)
        ax5.set_ylim(-2, 0)
        ax3.set_xlim(-0.1, 2.0)
        ax4.set_xlim(-2, 0)
        ax5.set_xlim(0, 1.5)

        # adjust
        subplots_adjust(hspace=0.2, wspace=1.0)

        # save
        savefig(@sprintf("images/anim%03d.png", i-1), dpi=150)
        println(@sprintf("images/anim%03d.png", i-1))
    end
end

# m, s = run_ridge()
# m, s = run_ridge(; bl=true)

pc = 1/6
setup_file = string(out_folder, "setup.h5")
state_files = string.(out_folder, "state", 0:300, ".h5")
plot_anim(setup_file, state_files)

println("Done.")