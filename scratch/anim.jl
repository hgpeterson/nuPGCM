using nuPGCM
using PyPlot
using PyCall
using Printf

gridspec = pyimport("matplotlib.gridspec")
slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function run_single_column(; bl=false)
    # parameters
    f = -5.5e-5
    N2 = 1e-6
    nz = 2^8
    H = 2e3
    θ = 2.5e-3         
    transport_constraint = false
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
    Δt = 10*secs_in_day
    t_save = 3*secs_in_year
    
    # create model struct
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transport_constraint, U)

    # save and log params
    save_setup(m)

    # set initial state
    b = zeros(nz)
    χ = zeros(nz)
    χ, u, v = invert(m, b, χ)
    i = [1]
    s = ModelState1DPG(b, χ, u, v, i)

    # solve transient
    evolve!(m, s, 3*secs_in_year, t_save) 
    
    return m, s
end

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
    Δt = 10*secs_in_day
    t_plot = 15*secs_in_year
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
    χ = zeros(nξ, nσ)
    uξ = zeros(nξ, nσ)
    uη = zeros(nξ, nσ)
    uσ = zeros(nξ, nσ)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, 3*secs_in_year, t_plot, t_save) 

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
        # savefig(@sprintf("images/anim%03d.png", i-1), dpi=200)
        savefig(@sprintf("images/anim%03d.png", i-1))
        println(@sprintf("images/anim%03d.png", i-1))
    end

    # then run:
    #   ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ../anim.mp4
end

function plot_can(setup_file_1D_can, state_file_1D_can, setup_file_2D, state_file_2D)
    # 2D
    m = load_setup_2D(setup_file_2D)
    ix = argmin(abs.(m.ξ .- m.L/4))
    s = load_state_2D(state_file_2D)

    # figure
    fig = figure(figsize=(25*pc, 26*pc))
    gs = gridspec.GridSpec(2, 6)
    ax1 = fig.add_subplot(get(gs, (0, slice(0, 3))))
    ax2 = fig.add_subplot(get(gs, (0, slice(3, 6))))
    ax3 = fig.add_subplot(get(gs, (1, slice(0, 2))))
    ax4 = fig.add_subplot(get(gs, (1, slice(2, 4))))
    ax5 = fig.add_subplot(get(gs, (1, slice(4, 6))))
    fig.suptitle(latexstring(L"$t = $", @sprintf("%d years", (s.i[1] - 1)*m.Δt/secs_in_year)))

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
    ax3.plot(1e3*s.χ[ix, :],  m.z[ix, :]/1e3, label="2D")
    ax4.plot(1e2*s.uη[ix, :], m.z[ix, :]/1e3, label="2D")
    ax5.plot(1e6*bz,          m.z[ix, :]/1e3, label="2D")

    # canonical
    m = load_setup_1D(setup_file_1D_can)
    s = load_state_1D(state_file_1D_can)
    bz = m.N2 .+ differentiate(s.b, m.z)
    ax3.plot(1e3*s.χ,  m.z/1e3, label="Canonical 1D")
    ax4.plot(1e2*s.v,  m.z/1e3, label="Canonical 1D")
    ax5.plot(1e6*bz,   m.z/1e3, label="Canonical 1D")

    # legend
    ax3.legend(ncol=2, loc=(1.0, 1.01))

    # limits
    ax3.set_ylim(-2, 0)
    ax4.set_ylim(-2, 0)
    ax5.set_ylim(-2, 0)
    ax3.set_xlim(-2, 50)
    ax3.set_xticks(0:25:50)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_xticks(-1.5:1.5:1.5)
    ax5.set_xlim(0, 1.5)

    # adjust
    subplots_adjust(hspace=0.4, wspace=1.0)

    # save
    # savefig("images/can.png")
    # println("images/can.png")
    savefig("images/can.pdf")
    println("images/can.pdf")
end

function plot_tc(setup_file_1D_tc, state_file_1D_tc, setup_file_1D_can, state_file_1D_can, setup_file_2D, state_file_2D)
    # 2D
    m = load_setup_2D(setup_file_2D)
    ix = argmin(abs.(m.ξ .- m.L/4))
    s = load_state_2D(state_file_2D)

    # figure
    fig = figure(figsize=(25*pc, 26*pc))
    gs = gridspec.GridSpec(2, 6)
    ax1 = fig.add_subplot(get(gs, (0, slice(0, 3))))
    ax2 = fig.add_subplot(get(gs, (0, slice(3, 6))))
    ax3 = fig.add_subplot(get(gs, (1, slice(0, 2))))
    ax4 = fig.add_subplot(get(gs, (1, slice(2, 4))))
    ax5 = fig.add_subplot(get(gs, (1, slice(4, 6))))
    fig.suptitle(latexstring(L"$t = $", @sprintf("%d years", (s.i[1] - 1)*m.Δt/secs_in_year)))

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
    ax3.plot(1e3*s.χ[ix, :],  m.z[ix, :]/1e3, label="2D")
    ax4.plot(1e2*s.uη[ix, :], m.z[ix, :]/1e3, label="2D")
    ax5.plot(1e6*bz,          m.z[ix, :]/1e3, label="2D")

    # canonical
    m = load_setup_1D(setup_file_1D_can)
    s = load_state_1D(state_file_1D_can)
    bz = m.N2 .+ differentiate(s.b, m.z)
    ax3.plot(1e3*s.χ,  m.z/1e3, label="Canonical 1D", zorder=0)
    ax4.plot(1e2*s.v,  m.z/1e3, label="Canonical 1D", zorder=0)
    ax5.plot(1e6*bz,   m.z/1e3, label="Canonical 1D", zorder=0)

    # tc
    m = load_setup_1D(setup_file_1D_tc)
    s = load_state_1D(state_file_1D_tc)
    z = @. m.H*(1 - cos(pi*(0:m.nz-1)/(m.nz-1)))/2
    χ, b = get_full_soln(m, s, z)
    δ, μ, S, q = get_BL_params(m)
    v = @. -m.f*s.χ[1]/q/m.ν[1] - tan(m.θ)/m.f*(s.b - s.b[1])
    bz = m.N2 .+ differentiate(b, z)
    ax3.plot(1e3*χ,  (z .- z[end])/1e3, c="tab:olive", ls=(0, (5, 5)), lw=1.0, label="Transport-constrained 1D BL")
    ax4.plot(1e2*v,  m.z/1e3,           c="tab:olive", ls=(0, (5, 5)), lw=1.0, label="Transport-constrained 1D BL")
    ax5.plot(1e6*bz, (z .- z[end])/1e3, c="tab:olive", ls=(0, (5, 5)), lw=1.0, label="Transport-constrained 1D BL")

    # legend
    ax3.legend(ncol=3, loc=(0.1, 1.01))

    # limits
    ax3.set_ylim(-2, 0)
    ax4.set_ylim(-2, 0)
    ax5.set_ylim(-2, 0)
    ax3.set_xlim(-0.1, 2)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_xticks(-1.5:1.5:1.5)
    ax5.set_xlim(0, 1.5)

    # adjust
    subplots_adjust(hspace=0.4, wspace=1.0)

    # save
    # savefig("images/tc.png")
    # println("images/tc.png")
    savefig("images/tc.pdf")
    println("images/tc.pdf")
end

# m, s = run_ridge()
# m, s = run_single_column()
# m, s = run_single_column(bl=true)

pc = 1/6

# setup_file = string(out_folder, "setup.h5")
# state_files = string.(out_folder, "state", 0:300, ".h5")
# plot_anim(setup_file, state_files)

# setup_file_1D_can = string(out_folder, "setup1Dcan.h5")
# state_file_1D_can = string.(out_folder, "state1Dcan.h5")
# setup_file_2D = string(out_folder, "setup2D.h5")
# state_file_2D = string.(out_folder, "state2D.h5")
# plot_can(setup_file_1D_can, state_file_1D_can, setup_file_2D, state_file_2D)

setup_file_1D_tc = string(out_folder, "setup1Dtc.h5")
state_file_1D_tc = string.(out_folder, "state1Dtc.h5")
setup_file_1D_can = string(out_folder, "setup1Dcan.h5")
state_file_1D_can = string.(out_folder, "state1Dcan.h5")
setup_file_2D = string(out_folder, "setup2D.h5")
state_file_2D = string.(out_folder, "state2D.h5")
plot_tc(setup_file_1D_tc, state_file_1D_tc, setup_file_1D_can, state_file_1D_can, setup_file_2D, state_file_2D)

println("Done.")