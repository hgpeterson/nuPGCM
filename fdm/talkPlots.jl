# for loading rotated data
include("rotatedCoords/rotated.jl")
include("spinDown/rotated.jl")
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

function canonicalSteadyTheory(z, κ0, κ1, h, N, f, θ, Pr)
    S = N^2*tan(θ)^2/f^2
    q = (f^2*cos(θ)^2*(1 + S*Pr)/(4*Pr*(κ0 + κ1)^2))^(1/4)
    Bz = @. N^2*cos(θ)*(κ0/(κ0 + κ1*exp(-z/h)) + κ1*exp(-z/h)/(κ0 + κ1*exp(-z/h))*S*Pr/(1 + S*Pr) - (κ0/(κ0 + κ1) + κ1/(κ0 + κ1)*S*Pr/(1 + S*Pr))*exp(-q*z)*(cos(q*z) + sin(q*z)))
    u = @. -κ1*cot(θ)*exp(-z/h)/h*S*Pr/(1 + S*Pr) + 2*q*cot(θ)*(κ0 + κ1*S*Pr/(1 + S*Pr))*exp(-q*z)*sin(q*z)
    vz = @. 1 # FIXME
    v = cumtrapz(vz, z)
    return Bz, u, v
end

function uvAnimation(folder)
    ix = 1
    tDays = 0:10:1000
    #= tDays = 1000 =#
    for tDay in tDays
        # setup plot
        fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2))

        ax[1].set_xlabel(L"cross-slope velocity, $u$ (m s$^{-1}$)")
        ax[1].set_ylabel(L"$z$ (m)")
        ax[1].set_xlim([-0.0005, 0.007])
        ax[1].set_ylim([z[ix, 1], z[ix, 1] + 200])
        ax[1].set_title(string(L"$t = $", tDay, " days"))

        ax[2].set_xlabel(L"along-slope velocity, $v$ (m s$^{-1}$)")
        ax[2].set_ylabel(L"$z$ (m)")
        ax[2].set_xlim([-0.03, 0.03])
        ax[2].set_title(string(L"$t = $", tDay, " days"))

        # full 2D solution
        b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/checkpoint", tDay, ".h5"))
        u, v, w = transformFromTF(uξ, uη, uσ)
        ax[1].plot(u[ix, :], z[ix, :], label="full 2D")
        ax[2].plot(v[ix, :], z[ix, :], label="full 2D")

        # canonical 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/checkpoint", tDay, ".h5"))
        u, w = rotate(û)
        ax[1].plot(u[ix, :], z[ix, :], label="canonical 1D")
        ax[2].plot(v[ix, :], z[ix, :], label="canonical 1D")

        # fixed 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "fixed1D/checkpoint", tDay, ".h5"))
        u, w = rotate(û)
        ax[1].plot(u[ix, :], z[ix, :], "k--", label="fixed 1D")
        ax[2].plot(v[ix, :], z[ix, :], "k--", label="fixed 1D")

        ax[1].legend(loc="upper right")
        tight_layout()
        fname = @sprintf("uvProfiles%04d.png", tDay)
        #= fname = @sprintf("uvProfiles%04d.pdf", tDay) =#
        savefig(fname, dpi=200)
        println(fname)
        close()
    end
end

function chivAnimation(folder)
    ix = 1
    tDays = 0:10:1000
    #= tDays = 1000 =#
    for tDay in tDays
        # setup plot
        fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)

        #= ax[1].set_xlabel(L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)") =#
        ax[1].set_xlabel(L"streamfunction, $\chi$ (rescaled)")
        ax[1].set_ylabel(L"$z$ (m)")
        #= ax[1].set_xlim([-0.005, 0.1]) =#
        ax[1].set_xlim([-0.1, 1.1])
        ax[1].set_xticks([0])
        ax[1].set_title(string(L"$t = $", tDay, " days"))

        ax[2].set_xlabel(L"along-slope velocity, $v$ (m s$^{-1}$)")
        ax[2].set_xlim([-0.03, 0.03])
        ax[2].set_title(string(L"$t = $", tDay, " days"))

        # zero line
        #= ax[1].axvline(0, lw=0.5, c="k", ls="-") =#

        # full 2D solution
        b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/checkpoint", tDay, ".h5"))
        u, v, w = transformFromTF(uξ, uη, uσ)
        #= ax[1].plot(chi[ix, :], z[ix, :], label="full 2D") =#
        c = maximum(chi[ix, :])
        if c == 0
            c = 1
        end
        ax[1].plot(chi[ix, :]/c, z[ix, :], label="full 2D")
        ax[2].plot(v[ix, :], z[ix, :], label="full 2D")

        # canonical 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/checkpoint", tDay, ".h5"))
        #= ax[1].plot(chi[ix, :], z[ix, :], label="canonical 1D") =#
        c = maximum(chi[ix, :])
        if c == 0
            c = 1
        end
        ax[1].plot(chi[ix, :]/c, z[ix, :], label="canonical 1D")
        ax[2].plot(v[ix, :], z[ix, :], label="canonical 1D")

        # fixed 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "fixed1D/checkpoint", tDay, ".h5"))
        #= ax[1].plot(chi[ix, :], z[ix, :], "k:", label="fixed 1D") =#
        c = maximum(chi[ix, :])
        if c == 0
            c = 1
        end
        ax[1].plot(chi[ix, :]/c, z[ix, :], "k:", label="fixed 1D")
        ax[2].plot(v[ix, :], z[ix, :], "k:", label="fixed 1D")

        ax[2].legend(loc="upper right")
        tight_layout()
        fname = @sprintf("chivProfilesRescaled%04d.png", tDay)
        #= fname = @sprintf("chivProfilesRescaled%04d.pdf", tDay) =#
        savefig(fname, dpi=200)
        println(fname)
        close()
    end
end

function idealRidge()
    fig, ax = subplots(1)
    ax.fill_between(x[:, 1]/1000, z[:, 1], minimum(z), color="k", alpha=0.3, lw=0.0)
    ax.annotate("", xy=(1500, -2800), xytext=(1500, 0), xycoords="data", arrowprops=Dict("arrowstyle" => "<->"))
    ax.annotate(L"$H(x)$", (1600, -1000), xycoords="data")
    ax.annotate(L"$κ(z)$", (1000, -800), xycoords="data")
    ax.annotate(L"$\bf{u}$", (1200, -2200), xycoords="data")
    ax.annotate("periodic", xy=(0, -400), xytext=(300, -450), xycoords="data", arrowprops=Dict("arrowstyle" => "<->"))
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_ylim([minimum(z), 0])
    ax.set_xlabel(L"$x$ (km)")
    ax.set_ylabel(L"$z$ (m)")
    tight_layout()
    savefig("ideal_ridge.svg")
end

function uBalance(folder)
    iξ = 1

    fig, ax = subplots(1)

    # full 2D
    b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/checkpoint1000.h5"))
    u, v, w = transformFromTF(uξ, uη, uσ)
    û = @. cosθ*u + sinθ*w
    ẑ = @. z*cosθ
    û_spl = Spline1D(ẑ[iξ, :], û[iξ, :]/maximum(û[iξ, :]))
    ẑ_plot = ẑ[iξ, 1] .+ 0:0.1:200
    #= ax.plot(û_spl(ẑ_plot), ẑ_plot, c="k") =#
    ax.plot(û_spl(ẑ_plot), ẑ_plot, label="full 2D")
    ax.fill_betweenx(ẑ_plot, û_spl(ẑ_plot), 0, alpha=0.5)

    b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/checkpoint1000.h5"))
    û_spl = Spline1D(ẑ[iξ, :], û[iξ, :]/maximum(û[iξ, :]))
    ẑ_plot = ẑ[iξ, 1] .+ 0:0.1:200
    #= ax.plot(û_spl(ẑ_plot), ẑ_plot, c="k") =#
    ax.plot(û_spl(ẑ_plot), ẑ_plot, label="canonical 1D")
    ax.fill_betweenx(ẑ_plot, û_spl(ẑ_plot), 0, alpha=0.5)

    ax.annotate(L"$\int \hat{u}$ d$\hat{z} = 0 \rightarrow$ far-field downwelling", xy=(0.04, 0.4), xytext=(0.2, 0.6), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    ax.annotate(L"$\kappa =$ const.", (0.7, 0.1), xycoords="axes fraction")

    #= ax.axvline(0, lw=1, c="k", ls="-") =#
    #= ax.set_ylim([ẑ[iξ, 1] - 10, ẑ[iξ, 1] + 200]) =#
    #= ax.set_xticks([]) =#
    #= ax.set_yticks([]) =#
    #= ax.spines["left"].set_visible(false) =#
    #= ax.spines["bottom"].set_visible(false) =#

    ax.set_xlabel(L"$\hat{u}$ (rescaled)")
    ax.set_ylabel(L"$\hat{z}$")
    ax.set_ylim([ẑ[iξ, 1], ẑ[iξ, 1] + 200])
    ax.set_xticks([0])
    ax.set_yticks([ẑ[iξ, 1], ẑ[iξ, 1] + 2*8.53])
    ax.set_yticklabels([0, L"2$\delta$"])
    ax.legend(loc="upper right")
    tight_layout()
    savefig("ubalance.pdf")
    #= savefig("ubalance.svg", transparent=true) =#
end

function chiBalance(folder)
    iξ = 1

    fig, ax = subplots(1)

    # full 2D
    b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/checkpoint1000.h5"))
    ax.plot(chi[iξ, :]/maximum(chi[iξ, :]), z[iξ, :], label="full 2D")

    b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/checkpoint1000.h5"))
    ax.plot(chi[iξ, :]/maximum(chi[iξ, :]), z[iξ, :], label="canonical 1D")

    ax.annotate(L"$U = \chi(0) = 0$", xy=(0.0, 0.99), xytext=(0.2, 0.95), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))
    ax.annotate("far-field downwelling", (0.2, 0.2), rotation=-20, xycoords="axes fraction")
    ax.annotate(L"$\kappa =$ const.", (0.1, 0.1), xycoords="axes fraction")

    ax.set_xlabel(L"$\chi$ (rescaled)")
    ax.set_xticks([0, U/maximum(chi[iξ, :])])
    ax.set_xticklabels([0, L"$\hat{U} > 0$"])
    ax.set_xlim([0, 1.1])
    ax.set_ylabel(L"$z$ (m)")
    ax.legend(loc=(0.45, 0.6))
    tight_layout()
    savefig("chibalance.pdf")
end

function chiForSketch(folder)
    iξ = 1

    fig, ax = subplots(1)

    # full 2D
    b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/checkpoint1000.h5"))
    ax.plot(chi[iξ, :]/maximum(chi[iξ, :]), z[iξ, :], "k", label="full 2D")

    b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/checkpoint1000.h5"))
    ax.plot(chi[iξ, :]/maximum(chi[iξ, :]), z[iξ, :], "k", label="canonical 1D")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(false)
    ax.axvline(0, ls="-", lw=0.5, c="k")
    ax.set_ylim([-H0, 0])

    tight_layout()
    savefig("chiForSketch.svg", transparent=true)
end

function chi_v_ridge(folder)
    # load
    c = loadCheckpointTF(string(folder, "2dpg/Pr1/checkpoint1000.h5"))
    u, v, w = transformFromTF(c.uξ, c.uη, c.uσ)

    # plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)
    ridgePlot(c.chi, c.b, "", L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)"; ax=ax[1])
    ridgePlot(v, c.b, "", L"along-ridge flow, $v$ (m s$^{-1}$)"; ax=ax[2])
    ax[1].annotate("(a)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (0.0, 1.05), xycoords="axes fraction")
    ax[2].set_ylabel("")
    savefig("chi_v_ridge.pdf")
    println("chi_v_ridge.pdf")
    close()
end

function profiles2Dvs1D(folder)
    ix = argmin(abs.(ξ .- L/4))
    tDays = 1000:1000:5000
    σ = 1 # prandtl number
    #= σ = 200 =# 
    
    # init plot
    fig, ax = subplots(3, 2, figsize=(3.404*2, 3*3.404/1.62), sharey=true)

    ax[1, 1].set_xlabel(L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)")
    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[1, 1].set_title("canonical 1D")

    ax[1, 2].set_xlabel(L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)")
    ax[1, 2].set_title(L"2D $\nu$PGCM")

    ax[2, 1].set_xlabel(L"along-ridge flow, $v$ (m s$^{-1}$)")
    ax[2, 1].set_ylabel(L"$z$ (km)")

    ax[2, 2].set_xlabel(L"along-ridge flow, $v$ (m s$^{-1}$)")

    ax[3, 1].set_xlabel(L"stratification, $B_z$ (s$^{-2}$)")
    ax[3, 1].set_ylabel(L"$z$ (km)")

    ax[3, 2].set_xlabel(L"stratification, $B_z$ (s$^{-2}$)")

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # plot data from folder
    for i=1:size(tDays, 1)
        tDay = tDays[i]

        # canonical 1D solution
        c = loadCheckpointRot(string(folder, "1dcan/Pr", σ, "/checkpoint", tDay, ".h5"))
        Bz = c.N^2*cos(c.θ[1]) .+ differentiate(c.b[1, :], c.ẑ[1, :].*cos(c.θ[1]))
        ax[1, 1].plot(c.chi[1, :],  c.ẑ[1, :]*cos(c.θ[1])/1e3, c=colors[i, :], label=string("Day ", Int64(tDay)))
        ax[2, 1].plot(c.v[1, :],    c.ẑ[1, :]*cos(c.θ[1])/1e3, c=colors[i, :])
        ax[3, 1].plot(Bz,           c.ẑ[1, :]*cos(c.θ[1])/1e3, c=colors[i, :])
        
        # 2D PG solution
        c = loadCheckpointTF(string(folder, "2dpg/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, v, w = transformFromTF(c.uξ, c.uη, c.uσ)
        Bz = c.N^2 .+ zDerivativeTF(c.b)
        ax[1, 2].plot(c.chi[ix, :], z[ix, :]/1e3, c=colors[i, :])
        ax[2, 2].plot(v[ix, :],     z[ix, :]/1e3, c=colors[i, :])
        ax[3, 2].plot(Bz[ix, :],    z[ix, :]/1e3, c=colors[i, :])

        # transport-constrained 1D solution
        c = loadCheckpointRot(string(folder, "1dtc/Pr", σ, "/checkpoint", tDay, ".h5"))
        Bz = c.N^2*cos(c.θ[1]) .+ differentiate(c.b[1, :], c.ẑ[1, :].*cos(c.θ[1]))
        ax[1, 2].plot(c.chi[1, :],  c.ẑ[1, :]*cos(c.θ[1])/1e3, c="k", ls=":")
        ax[2, 2].plot(c.v[1, :],    c.ẑ[1, :]*cos(c.θ[1])/1e3, c="k", ls=":")
        ax[3, 2].plot(Bz,           c.ẑ[1, :]*cos(c.θ[1])/1e3, c="k", ls=":")
    end

    ax[1, 1].legend(loc="center left")
    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["transport-constrained 1D"]
    ax[1, 2].legend(custom_handles, custom_labels, loc=(0.3, 0.4))

    ax[1, 1].annotate("(a)", (0.06, 0.92), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (0.06, 0.92), xycoords="axes fraction")
    ax[2, 1].annotate("(c)", (0.06, 0.92), xycoords="axes fraction")
    ax[2, 2].annotate("(d)", (0.06, 0.92), xycoords="axes fraction")
    ax[3, 1].annotate("(e)", (0.06, 0.92), xycoords="axes fraction")
    ax[3, 2].annotate("(f)", (0.06, 0.92), xycoords="axes fraction")

    ax[1, 2].annotate(string("Pr = ", σ), (0.8, 0.8), xycoords="axes fraction")

    tight_layout()
    savefig(string("profiles2Dvs1D_Pr", σ, ".pdf"))
    println(string("profiles2Dvs1D_Pr", σ, ".pdf"))
    close()
end

function BzChi2DvsFixed(folder)
    ix = 1
    tDays = 1000:1000:5000
    #= σ = 1 # prandtl number =#
    #= σ = 1000 =# 
    σ = 100 

    # setup plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2))
    ax[1].set_xlabel(L"stratification, $B_z$ (s$^{-2}$)")
    ax[1].set_ylabel(L"$z$ (m)")
    ax[2].set_xlabel(L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)")
    ax[2].set_ylabel(L"$z$ (m)")
    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # loop
    for i=1:size(tDays, 1)
        tDay = tDays[i]
        
        # full 2D solution
        b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, v, w = transformFromTF(uξ, uη, uσ)
        Bz = N^2 .+ zDerivativeTF(b)
        ax[1].plot(Bz[ix, :],  z[ix, :], c=colors[i, :], ls="-", label=string("Day ", Int64(tDay)))
        ax[2].plot(chi[ix, :], z[ix, :], c=colors[i, :], ls="-")

        # fixed 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "fixed1D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, w = rotate(û)
        Bz = N^2*cosθ[ix, :] .+ differentiate(b[ix, :], z[ix, :].*cosθ[ix, :])
        ax[1].plot(Bz,         z[ix, :], c="k", ls=":")
        ax[2].plot(chi[ix, :], z[ix, :], c="k", ls=":")
    end
    ax[1].legend(loc="upper left")
    tight_layout()
    fname = "BzChiPr100.pdf"
    savefig(fname)
    println(fname)
    close()
end

function uv2DvsFixed(folder)
    ix = 1
    tDays = 1000:1000:5000
    #= σ = 1 # prandtl number =#
    #= σ = 1000 =# 
    σ = 100 

    # setup plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2))
    ax[1].set_xlabel(L"cross-slope velocity, $u$ (m s$^{-1}$)")
    ax[1].set_ylabel(L"$z$ (m)")
    #= ax[1].set_ylim([z[ix, 1], z[ix, 1] + 1000]) =#
    ax[2].set_xlabel(L"along-slope velocity, $v$ (m s$^{-1}$)")
    ax[2].set_ylabel(L"$z$ (m)")
    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # loop
    for i=1:size(tDays, 1)
        tDay = tDays[i]
        
        # full 2D solution
        b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, v, w = transformFromTF(uξ, uη, uσ)
        ax[1].plot(u[ix, :],   z[ix, :], c=colors[i, :], ls="-", label=string("Day ", Int64(tDay)))
        ax[2].plot(v[ix, :],   z[ix, :], c=colors[i, :], ls="-")

        # fixed 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "fixed1D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, w = rotate(û)
        ax[1].plot(u[ix, :],   z[ix, :], c="k", ls=":")
        ax[2].plot(v[ix, :],   z[ix, :], c="k", ls=":")
    end
    ax[1].legend(loc="upper right")
    tight_layout()
    fname = "uvPr100.pdf"
    savefig(fname)
    println(fname)
    close()
end

function uvPrScaling(folder)
    ix = 1
    #= tDay = 1000 =#
    tDay = 5000
    Prs = [1, 10, 100, 1000]
    #= lss = ["-", "--", "-.", ":"] =#
    alphas = [1.0, 0.75, 0.5, 0.25]

    # setup plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2))
    ax[1].set_xlabel(L"cross-slope velocity, $u$ (m s$^{-1}$)")
    ax[1].set_ylabel(L"$z$ (m)")
    #= ax[1].set_xlim([-0.0005, 0.0035]) =#
    ax[1].set_ylim([z[ix, 1], z[ix, 1] + 200])
    ax[1].set_title(string(L"$t = $", tDay, " days"))
    ax[2].set_xlabel(L"along-slope velocity, $v$ (m s$^{-1}$)")
    ax[2].set_ylabel(L"$z$ (m)")
    #= ax[2].set_xlim([-0.015, 0.015]) =#
    ax[2].set_title(string(L"$t = $", tDay, " days"))

    # loop
    for i=1:size(Prs, 1)
        σ = Prs[i]
        ls = "-"
        alpha = alphas[i]

        # full 2D solution
        b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, v, w = transformFromTF(uξ, uη, uσ)
        ax[1].plot(u[ix, :], z[ix, :], c="tab:blue", ls=ls, label="full 2D", alpha=alpha)
        ax[2].plot(v[ix, :], z[ix, :], c="tab:blue", ls=ls, label="full 2D", alpha=alpha)

        # canonical 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, w = rotate(û)
        ax[1].plot(u[ix, :], z[ix, :], c="tab:orange", ls=ls, label="canonical 1D", alpha=alpha)
        ax[2].plot(v[ix, :], z[ix, :], c="tab:orange", ls=ls, label="canonical 1D", alpha=alpha)

        # fixed 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "fixed1D/Pr", σ, "/checkpoint", tDay, ".h5"))
        u, w = rotate(û)
        ax[1].plot(u[ix, :], z[ix, :], c="k", ls="--", label="fixed 1D", alpha=alpha)
        ax[2].plot(v[ix, :], z[ix, :], c="k", ls="--", label="fixed 1D", alpha=alpha)

        if i == 1
            ax[1].legend(loc="upper right")
        end
    end
    tight_layout()
    fname = "uvPrScaling.pdf"
    savefig(fname)
    println(fname)
    close()
end

function BzChiPrScaling(folder)
    ix = 1
    #= tDay = 1000 =#
    tDay = 5000
    Prs = [1, 10, 100, 1000]
    #= lss = ["-", "--", "-.", ":"] =#
    alphas = [1.0, 0.75, 0.5, 0.25]

    # setup plot
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2))
    ax[1].set_xlabel(L"stratification, $B_z$ (s$^{-2}$)")
    ax[1].set_ylabel(L"$z$ (m)")
    #= ax[1].set_xlim([-0.0005, 0.0035]) =#
    ax[1].set_title(string(L"$t = $", tDay, " days"))
    ax[1].ticklabel_format(style="sci", scilimits=(-4, 4))
    ax[2].set_xlabel(L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)")
    ax[2].set_ylabel(L"$z$ (m)")
    #= ax[2].set_xlim([-0.02, 0.18]) =#
    ax[2].set_title(string(L"$t = $", tDay, " days"))

    # loop
    for i=1:size(Prs, 1)
        σ = Prs[i]
        ls = "-"
        alpha = alphas[i]

        # full 2D solution
        b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(string(folder, "full2D/Pr", σ, "/checkpoint", tDay, ".h5"))
        Bz = N^2 .+ zDerivativeTF(b)
        ax[1].plot(Bz[ix, :], z[ix, :], c="tab:blue", ls=ls, label="full 2D", alpha=alpha)
        ax[2].plot(chi[ix, :], z[ix, :], c="tab:blue", ls=ls, label="full 2D", alpha=alpha)

        # canonical 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "canonical1D/Pr", σ, "/checkpoint", tDay, ".h5"))
        Bz = N^2*cosθ[ix, :] .+ differentiate(b[ix, :], z[ix, :].*cosθ[ix, :])
        ax[1].plot(Bz, z[ix, :], c="tab:orange", ls=ls, label="canonical 1D", alpha=alpha)
        ax[2].plot(chi[ix, :], z[ix, :], c="tab:orange", ls=ls, label="canonical 1D", alpha=alpha)

        # fixed 1D solution
        b, chi, û, v, U, t, L, H0, Pr, f, N, symmetry, κ = loadCheckpointRot(string(folder, "fixed1D/Pr", σ, "/checkpoint", tDay, ".h5"))
        Bz = N^2*cosθ[ix, :] .+ differentiate(b[ix, :], z[ix, :].*cosθ[ix, :])
        ax[1].plot(Bz, z[ix, :], c="k", ls="--", label="fixed 1D", alpha=alpha)
        ax[2].plot(chi[ix, :], z[ix, :], c="k", ls="--", label="fixed 1D", alpha=alpha)

        if i == 1
            ax[1].legend(loc="upper left")
        end
    end
    tight_layout()
    fname = "BzChiPrScaling.pdf"
    savefig(fname)
    println(fname)
    close()
end

function pressureRidgePlots(dfile)
    # read
    b, chi, uξ, uη, uσ, U, t, L, H0, Pr, f, N, symmetry, ξVariation, κ = loadCheckpointTF(dfile)

    # compute p_x
    u, v, w = transformFromTF(uξ, uη, uσ)
    px = f*v + zDerivativeTF(Pr*κ.*zDerivativeTF(u)) 

    # compute p
    p = zeros(size(b))
    p[:, end] = cumtrapz(px[:, end], ξ) # assume p = 0 at top left
    for i=1:nξ
        hydrostatic = H(ξ[i])*cumtrapz(b[i, :], σ)
        p[i, :] = hydrostatic .+ (p[i, end] - hydrostatic[end]) # integration constant from int(px)
    end

    ridgePlot(p, b, "pressure", L"$p$ (m$^2$ s$^{-2}$)"; cmap="viridis")
    savefig("p1000.pdf")
    close()
    ridgePlot(px, b, "pressure gradient", L"$p_x$ (m s$^{-2}$)")
    savefig("px1000.pdf")
    close()
end

function spindownProfiles(folder)
    # init plot
    fig, ax = subplots(3, 2, figsize=(3.404*2, 3*3.404/1.62), sharey=true)

    ax[1, 1].set_xlabel(L"cross-ridge flow, $\tilde{u}$")
    ax[1, 1].set_ylabel(L"$\tilde{z}$")
    ax[1, 1].set_title("canonical")

    ax[1, 2].set_xlabel(L"cross-ridge flow, $\tilde{u}$")
    ax[1, 2].set_title("transport-constrained")

    ax[2, 1].set_xlabel(L"along-ridge flow, $\tilde{v}$")
    ax[2, 1].set_ylabel(L"$\tilde{z}$")

    ax[2, 2].set_xlabel(L"along-ridge flow, $\tilde{v}$")

    ax[3, 1].set_xlabel(L"stratification, $\tilde{B}_\tilde{z}$")
    ax[3, 1].set_ylabel(L"$\tilde{z}$")

    ax[3, 2].set_xlabel(L"stratification, $\tilde{B}_\tilde{z}$")

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=5))

    # zoomed z
    ax[1, 1].set_ylim([0, 10])
    ax[2, 1].set_ylim([0, 10])
    ax[3, 1].set_ylim([0, 10])

    # plot data from folder
    cases = ["can", "tc"]
    for j=1:2
        case = cases[j]
        for i=0:5
            # load
            c = loadCheckpointSpinDown(string(folder, "/", case, "/checkpoint", i, ".h5"))
            τ_A = 1/c.S

            # stratification
            Bẑ = 1 .+ differentiate(c.b, c.ẑ)

            # colors and labels
            if i == 0
                color = "k"
                label = L"$\tilde{t} = 0$"
            else
                color = colors[i, :]
                label = string(L"$\tilde{t}/\tilde{\tau}_A$ = ", Int64(round(c.t/τ_A)))
            end

            # plot
            ax[1, j].plot(c.û,  c.ẑ, c=color, label=label)
            ax[2, j].plot(c.v,  c.ẑ, c=color, label=label)
            ax[3, j].plot(Bẑ,   c.ẑ, c=color, label=label)
            ax[2, j].axvline(c.Px, lw=1.0, c=color, ls="--")
        end
    end

    ax[1, 1].legend()

    ax[1, 1].annotate("(a)", (0.06, 0.92), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", (0.06, 0.92), xycoords="axes fraction")
    ax[2, 1].annotate("(c)", (0.06, 0.92), xycoords="axes fraction")
    ax[2, 2].annotate("(d)", (0.06, 0.92), xycoords="axes fraction")
    ax[3, 1].annotate("(e)", (0.06, 0.92), xycoords="axes fraction")
    ax[3, 2].annotate("(f)", (0.06, 0.92), xycoords="axes fraction")

    c = loadCheckpointSpinDown(string(folder, "/tc/checkpoint1.h5"))
    ax[1, 2].annotate(string(L"$\frac{\tilde{\tau}_A}{\tilde{\tau}_S} = $", @sprintf("%1.2f", 1/c.H/c.S)),  (0.6, 0.6), xycoords="axes fraction", size=10)
    #= ax[2, 2].annotate(L"$P_x$", xy=(0.05, 0.1), xytext=(0.2, 0.1), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->")) =#
    ax[2, 2].annotate(L"$P_x$", xy=(0.45, 0.1), xytext=(0.2, 0.1), xycoords="axes fraction", arrowprops=Dict("arrowstyle" => "->"))

    tight_layout()
    savefig("spindownProfiles.pdf")
    println("spindownProfiles.pdf")
    close()
end

function spindownGrid(folder)
    # read data
    file = h5open(string(folder, "vs.h5"), "r")
    vs = read(file, "vs")
    v0 = read(file, "v0")
    τ_Ss = read(file, "τ_Ss")
    τ_As = read(file, "τ_As")
    close(file)

    # text outline
    outline = [pe.withStroke(linewidth=0.6, foreground="k")]

    # plot grid
    fig, ax = subplots(1)
    ax.set_xlabel(L"spindown time, $\tilde{\tau}_S$")
    ax.set_ylabel(L"arrest time, $\tilde{\tau}_A$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    img = ax.pcolormesh(τ_Ss, τ_As, vs'/v0, rasterized=true, shading="auto", vmin=0, vmax=1)
    cb = colorbar(img, ax=ax, label=string("far-field along-slope flow,\n", L"$\tilde{v}/\tilde{v}_0$ at $\tilde{t} = 5\tilde{\tau}_A$"))
    ax.loglog([0, 1], [0, 1], transform=ax.transAxes, "w--", lw=0.5)
    ax.annotate(L"$\frac{\tilde{\tau}_A}{\tilde{\tau}_S} = 1$", xy=(0.7, 0.9), xycoords="axes fraction", c="w", path_effects=outline)
    ax.scatter(1e2, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax.scatter(5e3, 1e2, marker="o", facecolor="w", edgecolor="k", linewidths=0.5, zorder=2.5)
    ax.annotate("Fig. 6", xy=(0.3, 0.25), xycoords="axes fraction", c="w", path_effects=outline)
    ax.annotate("Fig. 7", xy=(0.8, 0.25), xycoords="axes fraction", c="w", path_effects=outline)

    ax.set_box_aspect(1)
    #= tight_layout() =#
    savefig("spindownGrid.pdf")
    println("spindownGrid.pdf")
end
