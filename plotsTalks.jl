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

function px3yr(folder)
    m = loadSetup2DPG(string(folder, "/const/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "/const/full2D/state1.h5"))

    # compute p_x
    u, v, w = transformFromTF(m, s)
    px = m.f*v + zDerivative(m, m.ν.*zDerivative(m, u)) 

    # # compute p
    # p = zeros(m.nξ, m.nσ)
    # p[:, end] = cumtrapz(px[:, end], m.ξ) # assume p = 0 at top left
    # for i=1:m.nξ
    #     hydrostatic = m.H[i]*cumtrapz(s.b[i, :], m.σ)
    #     p[i, :] = hydrostatic .+ (p[i, end] - hydrostatic[end]) # integration constant from int(px)
    # end

    ridgePlot(m, s, px, "", L"pressure gradient $\partial_x p$ (m s$^{-2}$)")
    savefig("px3yr.pdf")
    println("px3yr.pdf")
    close()
end

function chiProfile(folder)
    fig, ax = subplots(1, figsize=(3.404/1.62, 3.404))
    
    ax.set_xlabel(string(L"streamfunction $\chi$", "\n", L"($\times 10^{-3}$ m$^2$ s$^{-1}$)"))
    ax.set_ylabel(L"$\hat z$ (km)")

    c = loadCheckpoint1DTCPG(string(folder, "/1dtc_pg/tc/Pr1/checkpoint5.h5"))

    ax.plot(1e3*c.χ,  (c.ẑ .- c.ẑ[1])/1e3, lw=2)

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

    m = loadSetup2DPG(string(folder, "/const/bl2D/setup.h5"))
    s = loadState2DPG(string(folder, "/const/bl2D/state1.h5"))
    iξ = argmin(abs.(m.ξ .- m.L/4))

    χI = s.χ[iξ, :]
    bIξ = ξDerivative(m, s.b)
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
function boundaryCorrection(χI::Array{Float64,1}, z::Array{Float64,1}, q::Float64)
    A = -χI[1]
    χIz0 = differentiate_pointwise(χI[1:3], z[1:3], z[1], 1)
    B = -χIz0/q + A
    χB = @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
    return χB
end

function seamountBL1DFail(folder)
    # init plot
    fig, ax = subplots()

    ax.set_ylabel(L"$z$ (km)")
    ax.set_xlabel(string(L"stratification $\partial_z B$", "\n", L"($\times 10^{-6}$ s$^{-2}$)"))

    # model setups
    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))

    # limits
    ax.set_xlim([0, 1.2])
    ax.set_ylim([m.z[ix, 1]/1e3, m.z[ix, 1]/1e3 + 2])

    # plot data
    # BL 1D
    c = loadCheckpoint1DTCPG(string(folder, "L200km/bl1D/checkpoint5.h5"))
    ẑ = c.ẑ .- c.ẑ[1]
    q = (1/(4*c.Pr*c.κ[1])*(c.f^2*cos(c.θ)^2/(c.Pr*c.κ[1]) + c.N^2*sin(c.θ)^2/c.κ[1]))^(1/4)
    bI = c.b
    χI = -differentiate(bI, c.ẑ)*sin(c.θ)*c.Pr.*c.κ/(c.f^2*cos(c.θ)^2)
    χB = boundaryCorrection(χI, ẑ, q)
    bB = cumtrapz(χB*c.N^2*sin(c.θ)/c.κ[1], ẑ) .- trapz(χB*c.N^2*sin(c.θ)/c.κ[1], ẑ)
    χ = χI + χB
    b = bI + bB
    Bz = c.N^2*cos(c.θ) .+ differentiate(b, ẑ*cos(c.θ))
    v = cumtrapz(c.f*cos(c.θ)*(χ .- c.U)./(c.Pr*c.κ), ẑ)
    ax.plot(1e6*Bz, ẑ*cos(c.θ)/1e3 .+ m.z[ix, 1]/1e3, label="BL 1D")

    # full 2D
    s = loadState2DPG(string(folder, "L200km/full2D/state5.h5"))
    bz = differentiate(s.b[ix, :], m.z[ix, :])
    ax.plot(1e6*bz, m.z[ix, :]/1e3, "--", label="full 2D")

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
    m = loadSetup2DPG(string(folder, "L200km/full2D/setup.h5"))
    mBL = loadSetup2DPG(string(folder, "L200km/bl2D/setup.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ixBL = argmin(abs.(mBL.x[:, 1] .- mBL.L/4))

    # limits
    ax.set_xlim([0, 1.2])
    ax.set_ylim([m.z[ix, 1]/1e3, m.z[ix, 1]/1e3 + 2])

    # plot data
    # BL 2D
    s = loadState2DPG(string(folder, "L200km/bl2D/state5.h5"))
    bI = s.b[ixBL, :]
    χI = s.χ[ixBL, :]
    bIξ = ξDerivative(mBL, s.b)
    q = (1/(4*mBL.ν[ixBL, 1])*(mBL.f^2/mBL.ν[ixBL, 1] - mBL.Hx[ixBL]*bIξ[ixBL, 1]/mBL.H[ixBL]/mBL.κ[ixBL, 1]))^(1/4)
    χB = boundaryCorrection(χI, mBL.z[ixBL, :] .- mBL.z[ixBL, 1], q)
    bB = cumtrapz(χB*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) .- trapz(χB.*bIξ[ixBL, 1]/mBL.κ[ixBL, 1], mBL.z[ixBL, :]) 
    χ = χI + χB
    b = bI + bB
    bz = differentiate(b, mBL.z[ixBL, :])
    ax.plot(1e6*bz, mBL.z[ixBL, :]/1e3, label="BL 2D")

    # full 2D
    s = loadState2DPG(string(folder, "L200km/full2D/state5.h5"))
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

    m = loadSetup2DPG(string(folder, "/const/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "/const/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax)
    ax.plot([m.L/1e3/4, m.L/1e3/4], [m.z[ix, 1]/1e3, 0], "r-", alpha=0.5)

    tight_layout()

    savefig("chi3yr.pdf")
    println("chi3yr.pdf")
    plt.close()

    fig, ax = subplots(1)

    m = loadSetup2DPG(string(folder, "/exp/full2D/setup.h5"))
    s = loadState2DPG(string(folder, "/exp/full2D/state1.h5"))
    ix = argmin(abs.(m.x[:, 1] .- m.L/4))
    ridgePlot(m, s, s.χ, "", L"streamfunction $\chi$ (m$^2$ s$^{-1}$)"; ax=ax)
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

    mConst = loadSetup2DPG(string(folder, "/const/bl2D/setup.h5"))
    mExp   = loadSetup2DPG(string(folder, "/exp/bl2D/setup.h5"))

    s = loadState2DPG(string(folder, "/const/bl2D/state1.h5"))
    χtheory = BLtransport2D(mConst, s)
    W = exchangeVel2D(mConst, χtheory)
    ax[1].plot(mConst.ξ/1e3, 1e3*χtheory, label=string(L"$N^2 = $", "const."))
    ax[2].plot(mConst.ξ/1e3, 1e9*W)

    s = loadState2DPG(string(folder, "/exp/bl2D/state1.h5"))
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
function BLtransport2D(m::ModelSetup2DPG, s::ModelState2DPG)
    dbdξ = ξDerivative(m, s.b[:, 1])
    μ = m.ν[1, 1] / m.κ[1, 1]
    return @. m.κ[:, 1]/m.Hx * μ*m.Hx/m.f^2 * dbdξ / (1 - μ*m.Hx/m.f^2 * dbdξ)
end
function exchangeVel2D(m::ModelSetup2DPG, χ::Array{Float64,1})
    if m.coords == "cartesian"
        # uσ = -dξ(χ)/H
        uσ = -ξDerivative(m, χ)./m.H
    elseif m.coords == "cylindrical"
        # uσ = -dρ(ρ*χ)/(H*ρ)
        uσ = -ξDerivative(m, m.ξ.*χ)./m.H./m.ξ
        # assume χ = 0 at ρ = 0
        fd_ξ = mkfdstencil([0, m.ξ[1], m.ξ[2]], m.ξ[1], 1)
        uσ[1] = -(fd_ξ[2]*m.ξ[1]*χ[1] + fd_ξ[3]*m.ξ[2]*χ[2])/(m.H[1]*m.ξ[1])
    end
    return m.H.*uσ
end
path = "../sims/"

# spinupProfilesAnimation(string(path, "sim036/"))
# chiProfile(string(path, "sim026"))
px3yr(string(path, "sim037"))
# chiI_and_chiB(string(path, "sim037"))
# seamountBL1DFail(string(path, "sim035/"))
# seamountBL2DSuccess(string(path, "sim035/"))
# chi3yrN2exp(string(path, "sim037"))
# transportAndExchange(string(path, "sim037"))