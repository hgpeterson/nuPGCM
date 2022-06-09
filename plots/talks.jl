using nuPGCM, PyPlot, PyCall, Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# matplotlib
pl = pyimport("matplotlib.pylab")
pe = pyimport("matplotlib.patheffects")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")
pc = 1/6 # a pica is 1/6th of an inch

### utility functions

"""
    U = BLtransport2D(m, s)

Compute BL tranport from 2D BL theory.
"""
function BLtransport2D(m::ModelSetup2DPG, s::ModelState2DPG)
    dbdξ = ∂ξ(m, s.b[:, 1])
    μ = m.ν[1, 1] / m.κ[1, 1]
    return @. m.κ[:, 1]/m.Hx * μ*m.Hx/m.f^2 * dbdξ / (1 - μ*m.Hx/m.f^2 * dbdξ)
end

"""
    H*uσ = exchangeVel2D(m, χ)

Compute exchange velocity in 2D given BL transport (i.e. interior streamfunction at σ = -1).
""" 
function exchangeVel2D(m::ModelSetup2DPG, χ::Vector{Float64})
    if m.coords == "cartesian"
        # uσ = -dξ(χ)/H
        uσ = -∂ξ(m, χ)./m.H
    elseif m.coords == "cylindrical"
        # uσ = -dρ(ρ*χ)/(H*ρ)
        uσ = -∂ξ(m, m.ξ.*χ)./m.H./m.ξ
        # assume χ = 0 at ρ = 0
        fd_ξ = mkfdstencil([0, m.ξ[1], m.ξ[2]], m.ξ[1], 1)
        uσ[1] = -(fd_ξ[2]*m.ξ[1]*χ[1] + fd_ξ[3]*m.ξ[2]*χ[2])/(m.H[1]*m.ξ[1])
    end
    return m.H.*uσ
end

### plots

function spinupProfilesAnimation(folder)
    # plot data from folder
    m1dcan = loadSetup1DPG(string(folder, "1dcan/setup.h5"))
    m1dtc = loadSetup1DPG(string(folder, "1dtc/setup.h5"))
    m2d = load_setup_2DPG(string(folder, "2dpg/setup.h5"))
    for i=0:90
        # init plot
        fig, ax = subplots(1, 3, figsize=(6.5, 6.5/1.62/2), sharey=true)

        ax[1].set_ylabel(L"$z$ (km)")

        ax[1].set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
        ax[2].set_xlabel(string(L"along-slope flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
        ax[3].set_xlabel(string(L"stratification $\partial_z b$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

        ax[1].set_xlim([-5, 57])
        ax[2].set_xlim([-2.7, 1.4])
        ax[3].set_xlim([0, 1.3])
        ax[1].set_ylim([-2, 0])
        ax[2].set_ylim([-2, 0])
        ax[3].set_ylim([-2, 0])

        # canonical 1D solution
        s = loadState1DPG(string(folder, "1dcan/state", i, ".h5"))
        bz = m1dcan.N2 .+ differentiate(s.b, m1dcan.z)
        ax[1].plot(1e3*s.χ, m1dcan.z/1e3, label="canonical 1D")
        ax[2].plot(1e2*s.v, m1dcan.z/1e3, label="canonical 1D")
        ax[3].plot(1e6*bz,  m1dcan.z/1e3, label="canonical 1D")

        # transport-constrained 1D solution
        s = loadState1DPG(string(folder, "1dtc/state", i, ".h5"))
        bz = m1dtc.N2 .+ differentiate(s.b, m1dtc.z)
        ax[1].plot(1e3*s.χ, m1dtc.z/1e3, label="transport-\nconstrained 1D")
        ax[2].plot(1e2*s.v, m1dtc.z/1e3, label="transport-\nconstrained 1D")
        ax[3].plot(1e6*bz,  m1dtc.z/1e3, label="transport-\nconstrained 1D")
        
        # 2D PG solution
        s = load_state_2DPG(string(folder, "2dpg/state", i, ".h5"))
        ix = argmin(abs.(m2d.x[:, 1] .- m2d.L/4))
        v = s.uη
        bz = differentiate(s.b[ix, :], m2d.z[ix, :])
        ax[1].plot(1e3*s.χ[ix, :], m2d.z[ix, :]/1e3, "k:", label="2D")
        ax[2].plot(1e2*v[ix, :],   m2d.z[ix, :]/1e3, "k:", label="2D")
        ax[3].plot(1e6*bz,         m2d.z[ix, :]/1e3, "k:", label="2D")
        
        title = string(L"$t = $", Int64(round(s.i[1]*m2d.Δt/secsInYear)), " years")
        ax[2].set_title(title)
        ax[3].legend(loc="upper left")

        subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9, wspace=0.1, hspace=0.6)

        savefig(@sprintf("spinupProfiles%03d.png", i))
        println(@sprintf("spinupProfiles%03d.png", i))
        plt.close()
    end
end

function v3yr(folder)
    m = load_setup_2DPG(string(folder, "/const/full2D/setup.h5"))
    s = load_state_2DPG(string(folder, "/const/full2D/state1.h5"))

    # compute v
    u, v, w = transform_from_TF(m, s)

    ax = ridge_plot(m, s, 1e2*v, "", latexstring(L"along-ridge flow $v$", "\n", L"($\times 10^{-2}$ m s$^{-2}$)"); style="pcolormesh")
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ax.plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)
    savefig("v3yr.pdf")
    println("v3yr.pdf")
    plt.close()
end

function px3yr(folder)
    m = load_setup_2DPG(string(folder, "/const/full2D/setup.h5"))
    s = load_state_2DPG(string(folder, "/const/full2D/state1.h5"))

    # compute p_x
    u, v, w = transform_from_TF(m, s)
    px = m.f*v + ∂z(m, m.ν.*∂z(m, u)) 

    # # compute p
    # p = zeros(m.nξ, m.nσ)
    # p[:, end] = cumtrapz(px[:, end], m.ξ) # assume p = 0 at top left
    # for i=1:m.nξ
    #     hydrostatic = m.H[i]*cumtrapz(s.b[i, :], m.σ)
    #     p[i, :] = hydrostatic .+ (p[i, end] - hydrostatic[end]) # integration constant from int(px)
    # end

    ridge_plot(m, s, px, "", L"pressure gradient $\partial_x p$ (m s$^{-2}$)")
    savefig("px3yr.pdf")
    println("px3yr.pdf")
    plt.close()
end

function chiProfile(folder)
    fig, ax = subplots(1, figsize=(3.404/1.62, 3.404))
    
    ax.set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    ax.set_ylabel(L"$z$ (km)")

    m = loadSetup1DPG(string(folder, "/1dtc_pg/tc/mu1/setup.h5"))
    s = loadState1DPG(string(folder, "/1dtc_pg/tc/mu1/state1.h5"))

    ax.plot(1e3*s.χ,  (m.z .- m.z[1])/1e3, lw=2)

    # ax.set_xlim([0, 2.0])
    # ax.set_ylim([0, 0.5])

    tight_layout()

    savefig("chiProfile.pdf")
    println("chiProfile.pdf")
    plt.close()
end

function chiI_and_chiB(folder)
    fig, ax = subplots(1)
    
    ax.set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    ax.set_ylabel(L"$z$ (km)")

    m = load_setup_2DPG(string(folder, "/const/bl2D/setup.h5"))
    s = load_state_2DPG(string(folder, "/const/bl2D/state1.h5"))
    iξ = argmin(abs.(m.ξ .- m.L/4))

    χI = s.χ[iξ, :]
    bIξ = ∂ξ(m, s.b)
    q = (1/(4*m.ν[iξ, 1])*(m.f^2/m.ν[iξ, 1] - m.Hx[iξ]*bIξ[iξ, 1]/m.H[iξ]/m.κ[iξ, 1]))^(1/4)
    χB = boundaryCorrection(χI, m.z[iξ, :] .- m.z[iξ, 1], q)
    χ = χI + χB

    ax.plot(1e3*χ,  (m.z[iξ, :] .- m.z[iξ, 1])/1e3, "-",  lw=2, label=L"$\chi_I + \chi_B$")
    ax.plot(1e3*χI, (m.z[iξ, :] .- m.z[iξ, 1])/1e3, "--", lw=2, label=L"\chi_I")

    ax.set_xlim([0, 2.0])
    ax.set_ylim([0, 0.5])

    ax.legend()

    tight_layout()
    savefig("chiI_and_chiB.pdf")
    println("chiI_and_chiB.pdf")
    plt.close()
end

function seamountBL1DFail(folder)
    # init plot
    fig, ax = subplots()

    ax.set_ylabel(L"$z$ (km)")
    ax.set_xlabel(string(L"stratification $\partial_z B$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    m2D = load_setup_2DPG(string(folder, "full2D/setup.h5"))
    ix = argmin(abs.(m2D.x[:, 1] .- m2D.L/4))

    # limits
    ax.set_xlim([0, 1.6])

    # BL 1D
    m = loadSetup1DPG(string(folder, "bl1D/setup.h5"))
    s = loadState1DPG(string(folder, "bl1D/state5.h5"))
    z = m.z .- m.z[1]
    q = (1/(4*m.ν[1])*(m.f^2/m.ν[1] + m.N2*tan(m.θ)^2/m.κ[1]))^(1/4)
    bI = s.b
    χI = -differentiate(bI, m.z)*tan(m.θ).*m.ν/m.f^2
    χB = boundaryCorrection(χI, z, q)
    bB = cumtrapz(χB*m.N2*tan(m.θ)/m.κ[1], z) .- trapz(χB*m.N2*tan(m.θ)/m.κ[1], z)
    χ = χI + χB
    b = bI + bB
    Bz = m.N2 .+ differentiate(b, z)
    ax.plot(1e6*Bz, z/1e3 .+ m2D.z[ix, 1]/1e3, label="BL 1D")

    # full 2D
    s = load_state_2DPG(string(folder, "full2D/state5.h5"))
    bz = differentiate(s.b[ix, :], m2D.z[ix, :])
    ax.plot(1e6*bz, m2D.z[ix, :]/1e3, "--", label="full 2D")

    ax.legend()
    
    tight_layout()
    
    savefig("seamountBL1DFail.pdf")
    println("seamountBL1DFail.pdf")
    plt.close()
end

function seamountBL2DSuccess(folder)
    # init plot
    fig, ax = subplots()

    ax.set_ylabel(L"$z$ (km)")
    ax.set_xlabel(string(L"stratification $\partial_z B$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    # model setups
    m = load_setup_2DPG(string(folder, "full2D/setup.h5"))
    mBL = load_setup_2DPG(string(folder, "bl2D/setup.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ixBL = argmin(abs.(mBL.x[:, 1] .- mBL.L/4))

    # limits
    ax.set_xlim([0, 1.6])

    # plot data
    # BL 2D
    s = load_state_2DPG(string(folder, "bl2D/state5.h5"))
    bI = s.b[ixBL, :]
    χI = s.χ[ixBL, :]
    bIξ = ∂ξ(mBL, s.b)
    q = (1/(4*mBL.ν[ixBL, 1])*(mBL.f^2/mBL.ν[ixBL, 1] - mBL.Hx[ixBL]*bIξ[ixBL, 1]/mBL.H[ixBL]/mBL.κ[ixBL, 1]))^(1/4)
    χB = boundaryCorrection(χI, mBL.z[ixBL, :] .- mBL.z[ixBL, 1], q)
    bB = cumtrapz(χB*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) .- trapz(χB.*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) 
    χ = χI + χB
    b = bI + bB
    bz = differentiate(b, mBL.z[ixBL, :])
    ax.plot(1e6*bz, mBL.z[ixBL, :]/1e3, label="BL 2D")

    # full 2D
    s = load_state_2DPG(string(folder, "full2D/state5.h5"))
    bz = differentiate(s.b[ix, :], m.z[ix, :])
    ax.plot(1e6*bz, m.z[ix, :]/1e3, "--", label="full 2D")

    ax.legend()

    tight_layout()
    
    savefig("seamountBL2DSuccess.pdf")
    println("seamountBL2DSuccess.pdf")
    plt.close()
end

function chi3yrN2exp(folder)
    fig, ax = subplots(1)

    m = load_setup_2DPG(string(folder, "/const/full2D/setup.h5"))
    s = load_state_2DPG(string(folder, "/const/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridge_plot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax)
    ax.plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)

    tight_layout()

    savefig("chi3yr.pdf")
    println("chi3yr.pdf")
    plt.close()

    fig, ax = subplots(1)

    m = load_setup_2DPG(string(folder, "/exp/full2D/setup.h5"))
    s = load_state_2DPG(string(folder, "/exp/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridge_plot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax)
    ax.plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)

    tight_layout()

    savefig("chi3yrN2exp.pdf")
    println("chi3yrN2exp.pdf")
    plt.close()
end

function transportAndExchange(folder)
    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2)) 

    ax[1].set_xlabel(L"$\xi$ (km)")
    ax[1].set_ylabel(string("BL transport\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))

    ax[2].set_xlabel(L"$\xi$ (km)")
    ax[2].set_ylabel(string(L"exchange velocity $H u^\sigma$", "\n", L"($\times 10^{-9}$ m s$^{-1}$)"))

    ax[1].annotate("(a)", (-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", (-0.04, 1.05), xycoords="axes fraction")

    mConst = load_setup_2DPG(string(folder, "/const/bl2D/setup.h5"))
    mExp   = load_setup_2DPG(string(folder, "/exp/bl2D/setup.h5"))

    s = load_state_2DPG(string(folder, "/const/bl2D/state1.h5"))
    χtheory = BLtransport2D(mConst, s)
    W = exchangeVel2D(mConst, χtheory)
    ax[1].plot(mConst.ξ/1e3, 1e3*χtheory, label=string(L"$N^2 = $", "const."))
    ax[2].plot(mConst.ξ/1e3, 1e9*W)

    s = load_state_2DPG(string(folder, "/exp/bl2D/state1.h5"))
    χtheory = BLtransport2D(mExp, s)
    W = exchangeVel2D(mExp, χtheory)
    ax[1].plot(mExp.ξ/1e3, 1e3*χtheory,label=L"$N^2 \sim \exp(z/\delta)$")
    ax[2].plot(mExp.ξ/1e3, 1e9*W)
    

    ax[1].set_xlim([0, mConst.L/1e3])
    ax[1].set_ylim([-3, 3])

    ax[2].set_xlim([0, mConst.L/1e3])
    ax[2].set_ylim([-8, 18])

    ax[1].legend()

    tight_layout()

    savefig("transportAndExchange.pdf")
    println("transportAndExchange.pdf")
    plt.close()
end

function ridgeAnimation(folder)
    m = load_setup_2DPG(string(folder, "setup.h5"))
    for i=0:90
        s = load_state_2DPG(string(folder, "state$i.h5"))
        ax = ridge_plot(m, s, 1e3*s.χ, "", string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"); vext=1.5)
        ax.set_title(string(L"$t = $", @sprintf("%1.1f years", m.Δt*s.i[1]/secsInYear)))
        tight_layout()
        savefig(@sprintf("ridgeAnimation%03d.png", i))
        println(@sprintf("ridgeAnimation%03d.png", i))
        plt.close()
    end
end

function spinup_profiles_tc(folder; μ=1)
    ii = 1:5

    # init plot
    fig, ax = subplots(1, 3, figsize=(27*pc, 11.5*pc), sharey=true)

    ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")

    ax[1].set_xlabel(string(L"Streamfunction $\chi$", "\n", L"($\times10^{-3}$ m$^2$ s$^{-1}$)"))
    ax[2].set_xlabel(string(L"Along-ridge flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))
    ax[3].set_xlabel(string(L"Stratification $\partial_z B$", "\n", L"($\times10^{-6}$ s$^{-2}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    if μ == 1
        ax[1].set_xlim([-0.5, 1.65])
        ax[1].set_xticks(-0.5:0.5:1.5)
        ax[2].set_xlim([-3.0, 1.7])
        ax[2].set_xticks(-3:1.5:1.5)
        ax[3].set_xlim([0, 1.3])
    elseif μ == 200
        ax[1].set_xlim([-25, 95])
        ax[1].set_xticks(-25:25:75)
        ax[2].set_xlim([-2.0, 0.3])
        ax[3].set_xlim([0, 1.3])
    end

    # fixed y
    ax[1].set_ylim([-2, 0])

    # setup file
    m2D = load_setup_2D(string(folder, "2dpg/mu", μ, "/setup.h5"))

    # plot data from folder
    for i=ii
        # transport-constrained 1D solution
        m = load_setup_1D(string(folder, "1dtc_pg/tc/mu", μ, "/setup.h5"))
        s = load_state_1D(string(folder, "1dtc_pg/tc/mu", μ, "/state", i, ".h5"))
        label = string(Int64(m.Δt*s.i[1]/secs_in_year), " years")
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax[1].plot(1e3*s.χ, m.z/1e3, c=colors[i, :], label=label)
        ax[2].plot(1e2*s.v, m.z/1e3, c=colors[i, :], label=label)
        ax[3].plot(1e6*Bz,  m.z/1e3, c=colors[i, :], label=label)

        # 2D PG solution
        s2D = load_state_2D(string(folder, "2dpg/mu", μ, "/state", i, ".h5"))
        ix = argmin(abs.(m2D.x[:, 1] .- m2D.L/4))
        Bz2D = differentiate(s2D.b[ix, :], m2D.z[ix, :])
        ax[1].plot(1e3*s2D.χ[ix, :],  m2D.z[ix, :]/1e3, "k--", lw=0.5)
        ax[2].plot(1e2*s2D.uη[ix, :], m2D.z[ix, :]/1e3, "k--", lw=0.5)
        ax[3].plot(1e6*Bz2D,          m2D.z[ix, :]/1e3, "k--", lw=0.5)
    end

    ax[3].legend(loc="upper left")
    custom_handles = [lines.Line2D([0], [0], c="k", ls="--", lw=0.5)]
    custom_labels = ["2D"]
    ax[2].legend(custom_handles, custom_labels)

    subplots_adjust(hspace=0.4)
    # savefig(string("spinup_profiles_tc.pdf"))
    # println(string("spinup_profiles_tc.pdf"))
    savefig(string("spinup_profiles_tc.png"))
    println(string("spinup_profiles_tc.png"))
    plt.close()
end

function spinup_profiles_v(folder; μ=1)
    ii = 1:5

    # init plot
    fig, ax = subplots(1, 1, figsize=(9*pc, 10*pc), sharey=true)

    ax.set_ylabel(L"Vertical coordinate $z$ (km)")

    ax.set_xlabel(string(L"Along-ridge flow $v$", "\n", L"($\times10^{-2}$ m s$^{-1}$)"))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(ii, 1)))

    # fixed x
    if μ == 1
        ax.set_xlim([-3.0, 1.7])
        ax.set_xticks(-3:1.5:1.5)
    elseif μ == 200
        ax.set_xlim([-2.0, 0.3])
    end

    # fixed y
    ax.set_ylim([-2, 0])

    # setup file
    m2D = load_setup_2D(string(folder, "2dpg/mu", μ, "/setup.h5"))

    # plot data from folder
    for i=ii
        # canonical 1D solution
        m = load_setup_1D(string(folder, "1dtc_pg/can/mu", μ, "/setup.h5"))
        s = load_state_1D(string(folder, "1dtc_pg/can/mu", μ, "/state", i, ".h5"))
        label = string(Int64(m.Δt*s.i[1]/secs_in_year), " years")
        Bz = m.N2 .+ differentiate(s.b, m.z)
        ax.plot(1e2*s.v, m.z/1e3, c=colors[i, :], label=label, zorder=0)

        # 2D PG solution
        s2D = load_state_2D(string(folder, "2dpg/mu", μ, "/state", i, ".h5"))
        ix = argmin(abs.(m2D.x[:, 1] .- m2D.L/4))
        Bz2D = differentiate(s2D.b[ix, :], m2D.z[ix, :])
        ax.plot(1e2*s2D.uη[ix, :], m2D.z[ix, :]/1e3, "k--", lw=0.5)
    end

    # steady state canonical
    m = load_setup_1D(string(folder, "1dtc_pg/can/mu", μ, "/setup.h5"))
    s = load_state_1D(string(folder, "1dtc_pg/can/mu", μ, "/state-1.h5"))
    Bz = m.N2 .+ differentiate(s.b, m.z)
    ax.plot(1e2*s.v,  m.z/1e3, c="k", label="Steady state")

    ax.legend(loc=(0.8, 0.4))

    fig.text(0.55, 0.97, "1D", size=8, ha="left", va="top")
    fig.text(0.3, 0.97, "2D", size=8, ha="left", va="top")

    # savefig(string("spinup_profiles_v.pdf"))
    # println(string("spinup_profiles_v.pdf"))
    savefig(string("spinup_profiles_v.png"))
    println(string("spinup_profiles_v.png"))
    plt.close()
end

path = "../../sims/"

# spinupProfilesAnimation(string(path, "sim036/"))
# chiProfile(string(path, "sim039"))
# v3yr(string(path, "sim037"))
# px3yr(string(path, "sim037"))
# chiI_and_chiB(string(path, "sim037"))
# seamountBL1DFail(string(path, "sim042/"))
# seamountBL2DSuccess(string(path, "sim042/"))
# chi3yrN2exp(string(path, "sim037"))
# transportAndExchange(string(path, "sim037"))
# ridgeAnimation(string(path, "sim041/"))
spinup_profiles_tc(string(path, "sim039/"))
spinup_profiles_v(string(path, "sim039/"))