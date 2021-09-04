using PyPlot, PyCall, Printf, HDF5, Dierckx

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

include("myJuliaLib.jl")

# for loading data
include("1dtc/utils.jl")
include("1dtc_pg/utils.jl")
include("1dtc_nondim/utils.jl")
include("2dpg/setup.jl")
include("2dpg/utils.jl")
include("rayleigh/2dpg/utils.jl")
include("rayleigh/1dtc_pg/utils.jl")

# for ridgePlot
include("2dpg/plotting.jl")

# matplotlib
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

function spinupProfilesAnimation(folder)
    # plot data from folder
    for i=0:90
        # init plot
        fig, ax = subplots(1, 3, figsize=(6.5, 6.5/1.62/2), sharey=true)

        ax[1].set_ylabel(L"$z$ (km)")

        ax[1].set_xlabel(string(L"streamfunction, $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
        ax[2].set_xlabel(string(L"along-ridge flow, $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
        ax[3].set_xlabel(string(L"stratification, $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

        ax[1].set_xlim([-5, 57])
        ax[2].set_xlim([-2.7, 1.4])
        ax[3].set_xlim([0, 1.3])

        # canonical 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dcan/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, label="canonical 1D")
        ax[2].plot(1e2*c.v̂, c.ẑ*cos(c.θ)/1e3, label="canonical 1D")
        ax[3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, label="canonical 1D")

        # transport-constrained 1D solution
        c = loadCheckpoint1DTCPG(string(folder, "1dtc/checkpoint", i, ".h5"))
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ.*cos(c.θ))
        ax[1].plot(1e3*c.χ, c.ẑ*cos(c.θ)/1e3, label="transport-\nconstrained 1D")
        ax[2].plot(1e2*c.v̂,c.ẑ*cos(c.θ)/1e3, label="transport-\nconstrained 1D")
        ax[3].plot(1e6*Bz,  c.ẑ*cos(c.θ)/1e3, label="transport-\nconstrained 1D")
        
        # 2D PG solution
        m = loadSetup2DPG(string(folder, "2dpg/setup.h5"))
        s = loadState2DPG(string(folder, "2dpg/state", i, ".h5"))
        ix = argmin(abs.(m.x[:, 1] .- m.L/4))
        v = s.uη
        Bz = differentiate(s.b[ix, :], m.z[ix, :])
        ax[1].plot(1e3*s.χ[ix, :], m.z[ix, :]/1e3, "k:", label="2D")
        ax[2].plot(1e2*v[ix, :],   m.z[ix, :]/1e3, "k:", label="2D")
        ax[3].plot(1e6*Bz,         m.z[ix, :]/1e3, "k:", label="2D")
        
        title = string(L"$t = $", Int64(round(c.t/86400/360)), " years")
        ax[2].set_title(title)
        ax[3].legend(loc="upper left")

        subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

        savefig(@sprintf("spinupProfiles%03d.png", i))
        println(@sprintf("spinupProfiles%03d.png", i))
        plt.close()
    end

end

path = "../sims/"

spinupProfilesAnimation(string(path, "sim036/"))