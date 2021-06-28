pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
lines = pyimport("matplotlib.lines")

"""
    profilePlot(datafiles)

Plot profiles of b, χ, û, and v̂ from HDF5 snapshot files of buoyancy in the `datafiles` list.
"""
function profilePlot(datafiles)
    # init plot
    fig, ax = subplots(2, 2, figsize=(6.5, 6.5/1.62))

    # insets
    axins21 = inset_locator.inset_axes(ax[2, 1], width="40%", height="40%")

    ax[1, 1].set_xlabel(L"$B_z$ (s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$z$ (km)")
    ax[1, 1].set_title("stratification")

    ax[1, 2].set_xlabel(L"$\chi$ (m$^2$ s$^{-1}$)")
    ax[1, 2].set_ylabel(L"$z$ (km)")
    ax[1, 2].set_title("streamfunction")

    ax[2, 1].set_xlabel(L"$u$ (m s$^{-1}$)")
    ax[2, 1].set_ylabel(L"$z$ (km)")
    ax[2, 1].set_title("cross-ridge velocity")

    ax[2, 2].set_xlabel(L"$v$ (m s$^{-1}$)")
    ax[2, 2].set_ylabel(L"$z$ (km)")
    ax[2, 2].set_title("along-ridge velocity")

    subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end
    axins21.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)-1))

    # zoomed z
    ax[2, 1].set_ylim([z[1]/1e3, (z[1] + 2e2)/1e3])

    # plot data from `datafiles`
    for i=1:size(datafiles, 1)
        # load
        c = loadCheckpoint1DTCPG(datafiles[i])

        # convert to physical coordinates 
        u, w = rotate(c.û)

        # stratification
        Bz = c.N^2*cos(θ) .+ differentiate(c.b, ẑ)

        # colors and labels
        if c.t == -42
            # steady state
            label = "steady state"
            color = "k"
        else
            label = string(Int64(round(c.t/secsInYear)), " years")
            if i==1
                color = "k"
            else
                color = colors[i-1, :]
            end
        end

        # plot
        ax[1, 1].plot(Bz,       z/1e3, c=color, label=label)
        ax[1, 2].plot(c.χ,      z/1e3, c=color, label=label)
        ax[1, 2].axvline(c.U,          c=color, lw=1.0, ls="--")
        ax[2, 1].plot(u,        z/1e3, c=color, label=label)
        ax[2, 2].plot(c.v̂,      z/1e3, c=color, label=label)
        axins21.plot(u,         z/1e3, c=color, label=label)
    end

    ax[1, 2].legend()

    savefig("profiles.png")
    println("profiles.png")
end

"""
    profilePlotBL(datafilesFull, datafilesBL)

Compare profiles of b from HDF5 snapshot files of buoyancy in the `datafilesFull` and `datafilesBL` lists.
"""
function profilePlotBL(datafilesFull, datafilesBL)
    # init plot
    fig, ax = subplots(2, 3, figsize=(6.5, 4))

    ax[1, 1].set_xlabel(L"buoyancy, $b$ (m s$^{-2}$)")
    ax[1, 1].set_ylabel(L"$\hat z$ (km)")

    ax[1, 2].set_xlabel(L"stratification, $\partial_{\hat z} B$ (s$^{-2}$)")

    ax[1, 3].set_xlabel(L"streamfunction, $\chi$ (m$^2$ s$^{-1}$)")

    ax[2, 1].set_xlabel(L"BL buoyancy, $b$ (m s$^{-2}$)")
    ax[2, 1].set_ylabel(L"$\hat z$ (km)")

    ax[2, 2].set_xlabel(L"BL stratification, $\partial_{\hat z} B$ (s$^{-2}$)")

    ax[2, 3].set_xlabel(L"BL streamfunction, $\chi$ (m$^2$ s$^{-1}$)")

    subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.1, hspace=0.6)

    c = loadCheckpoint1DTCPG(datafilesBL[1])
    ax[1, 1].annotate(string(L"\sigma =", @sprintf("%1.2e", c.Pr)),                   (0.5, 0.6), xycoords="axes fraction")
    ax[1, 1].annotate(string(L"S =",      @sprintf("%1.2e", c.N^2*tan(c.θ)^2/c.f^2)), (0.5, 0.5), xycoords="axes fraction")

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    end
    for col=2:3
        ax[1, col].set_yticklabels([])
        ax[2, col].set_yticklabels([])
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafilesFull, 1)-1))

    # limits
    ax[2, 1].set_ylim([-c.H/1e3, -c.H/1e3 + 0.1])
    ax[2, 2].set_ylim([-c.H/1e3, -c.H/1e3 + 0.1])
    ax[2, 3].set_ylim([-c.H/1e3, -c.H/1e3 + 0.1])
    ax[2, 2].set_xlim([-1e-7, c.N^2/1.5])

    # plot data
    for i=1:size(datafilesFull, 1)
        # load
        c = loadCheckpoint1DTCPG(datafilesFull[i])
        cBL = loadCheckpoint1DTCPG(datafilesBL[i])

        # compute full BL solution
        S = cBL.N^2*tan(cBL.θ)^2/cBL.f^2
        bI = cBL.b
        bB = get_bB(bI, cBL.ẑ, cBL.f, cBL.θ, cBL.Pr, S, cBL.Pr*cBL.κ, cBL.H)
        bBL = bI + bB
        χI = -differentiate(bI, cBL.ẑ)*sin(cBL.θ)*cBL.Pr.*cBL.κ/(cBL.f^2*cos(cBL.θ)^2)
        χB = cBL.κ[1]/cBL.N^2/sin(cBL.θ)*differentiate(bB, cBL.ẑ)
        χBL = χI + χB

        # stratification
        Bz = c.N^2*cos(c.θ) .+ differentiate(c.b, c.ẑ)
        BzBL = cBL.N^2*cos(cBL.θ) .+ differentiate(bBL, cBL.ẑ)

        # colors and labels
        if c.t == -42
            # steady state
            label = "steady state"
            color = "k"
        else
            label = string(Int64(round(c.t/secsInYear)), " years")
            if i==1
                color = "k"
            else
                color = colors[i-1, :]
            end
        end

        # plot
        ax[1, 1].plot(c.b,   c.ẑ/1e3, c=color, label=label)
        ax[2, 1].plot(c.b,   c.ẑ/1e3, c=color, label=label)
        ax[1, 2].plot(Bz,    c.ẑ/1e3, c=color, label=label)
        ax[2, 2].plot(Bz,    c.ẑ/1e3, c=color, label=label)
        ax[1, 3].plot(c.χ,   c.ẑ/1e3, c=color, label=label)
        ax[2, 3].plot(c.χ,   c.ẑ/1e3, c=color, label=label)
        ax[1, 1].plot(bBL,   cBL.ẑ/1e3, c="k", ls=":")
        ax[2, 1].plot(bBL,   cBL.ẑ/1e3, c="k", ls=":")
        ax[1, 2].plot(BzBL,  cBL.ẑ/1e3, c="k", ls=":")
        ax[2, 2].plot(BzBL,  cBL.ẑ/1e3, c="k", ls=":")
        ax[1, 3].plot(χBL,   cBL.ẑ/1e3, c="k", ls=":")
        ax[2, 3].plot(χBL,   cBL.ẑ/1e3, c="k", ls=":")
    end

    custom_handles = [lines.Line2D([0], [0], c="k", ls=":", lw="1")]
    custom_labels = ["BL theory"]
    ax[1, 1].legend(custom_handles, custom_labels)
    ax[1, 2].legend()
    
    savefig("profilesBL.png")
    println("profilesBL.png")
end

function buoyancyFlux(datafile)
    c = loadCheckpoint1DTCPG(datafile)

    S = c.N^2*tan(c.θ)^2/c.f^2

    b = c.b
    b′= (1 + c.Pr*S)*c.b

    bz = differentiate(b, c.ẑ)
    bz′ = differentiate(b′, c.ẑ)

    Bz = c.N^2*cos(c.θ) .+ bz
    Bz′ = c.N^2*cos(c.θ) .+ bz′

    bt = differentiate(c.κ.*Bz, c.ẑ)
    bt′ = differentiate(c.κ.*Bz′, c.ẑ)

    fig, ax = subplots(1)
    ax.set_xlabel(L"stratification, $\partial_{\hat z} B$ (s$^{-2}$)")
    ax.set_ylabel(L"$\hat z$ (km)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.plot(Bz,  c.ẑ/1e3, label="pure diffusion")        
    ax.plot(Bz′, c.ẑ/1e3, label=L"$\sigma S$ correction")
    ax.legend()
    tight_layout()
    savefig("Bz.png")
    println("Bz.png")

    fig, ax = subplots(1)
    ax.set_xlabel(L"flux, $-\kappa \partial_{\hat z} B$ (m$^2$ s$^{-3}$)")
    ax.set_ylabel(L"$\hat z$ (km)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.plot(-c.κ.*Bz,  c.ẑ/1e3, label="pure diffusion")        
    ax.plot(-c.κ.*Bz′, c.ẑ/1e3, label=L"$\sigma S$ correction")
    ax.legend()
    tight_layout()
    savefig("F.png")
    println("F.png")

    fig, ax = subplots(1)
    ax.set_xlabel(L"convergence of flux, $\partial_{\hat z}[\kappa \partial_{\hat z} B]$ (m s$^{-3}$)")
    ax.set_ylabel(L"$\hat z$ (km)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.plot(bt,  c.ẑ/1e3, label="pure diffusion")        
    ax.plot(bt′, c.ẑ/1e3, label=L"$\sigma S$ correction")
    ax.legend()
    tight_layout()
    savefig("bt.png")
    println("bt.png")
end

"""
    bB = get_bB(bI, ẑ, f, θ, Pr, S, ν, H)

Compute boundary correction to interior solution `bI`.
"""
function get_bB(bI, ẑ, f, θ, Pr, S, ν, H)
    z = ẑ .+ H
    q = (f^2*cos(θ)^2*(1 + Pr*S)/4/ν[1]^2)^(1/4)
    bIz0 = differentiate_pointwise(bI[1:3], z[1:3], 0, 1)
    bIzz0 = differentiate_pointwise(bI[1:5], z[1:5], 0, 2)
    #= B = Pr*S*bIzz0/(2*q^2) =#
    #= A = -Pr*S*bIz0/q - B =#
    B = 0
    A = -Pr*S*bIz0/q
    return @. exp(-q*z)*(A*cos(q*z) + B*sin(q*z))
end
