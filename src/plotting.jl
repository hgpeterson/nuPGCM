"""
    quick_plot(u::AbstractVector, m::Mesh; b=nothing, label="", fname="image.png")

Plot a scalar field `u` on a mesh `m`.
"""
function quick_plot(u::AbstractVector, m::Mesh; b=nothing, label="", fname="image.png")
    # get aspect ratio right
    xL = minimum(m.p[:, 1])
    xR = maximum(m.p[:, 1])
    L = xR - xL
    yB = minimum(m.p[:, 2])
    yT = maximum(m.p[:, 2])
    H = yT - yB
    γ = H/L

    # plot
    fig, ax = plt.subplots(1, figsize=(3.2, 3.2*γ))
    umax = maximum(abs.(u))
    img = ax.tripcolor(m.p[:, 1], m.p[:, 2], m.t[:, 1:3] .- 1, u, shading="gouraud", vmin=-umax, vmax=umax, cmap="RdBu_r", rasterized=true)
    if b !== nothing
        b = unpack_fefunction(b, m)
        ax.tricontour(m.p[:, 1], m.p[:, 2], m.t[:, 1:3] .- 1, b, colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    end
    cb = plt.colorbar(img, ax=ax, label=label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.set_xticks([xL, xR])
    ax.set_yticks([yB, yT])
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig(fname)
    println(fname)
    plt.close()
end
function quick_plot(u::FEFunction, m::Mesh; kwargs...)
    u = unpack_fefunction(u, m)
    quick_plot(u, m; kwargs...)
end

function plot_yslice(u, b, y, H; t=nothing, cb_label="", fname="yslice.png")
    nx = 2^5
    nσ = 2^6
    σ = (chebyshev_nodes(nσ) .- 1)/2
    σσ = repeat(σ', nx, 1)
    x = range(-1, 1, length=nx)
    xx = repeat(x, 1, nσ)
    HH = [H([xx[i, j], y]) for i ∈ 1:nx, j ∈ 1:nσ]
    zz = σσ.*HH

    @time us = [nan_eval(u, Point(xx[i, j], y, zz[i, j])) for i ∈ 1:nx, j ∈ 1:nσ]
    @time bs = [nan_eval(b, Point(xx[i, j], y, zz[i, j])) for i ∈ 1:nx, j ∈ 1:nσ]
    us[:, 1] .= 0

    fig, ax = plt.subplots(1)
    img = ax.pcolormesh(xx, zz, us, shading="gouraud", cmap="RdBu_r", rasterized=true)
    plt.colorbar(img, ax=ax, label=cb_label)
    ax.contour(xx, zz, bs, colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.axis("equal")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    if t === nothing
        ax.set_title("Slice at "*L"y = "*@sprintf("%1.2f", y))
    else
        ax.set_title("Slice at "*L"y = "*@sprintf("%1.2f", y)*L", \; t = "*@sprintf("%1.2f", t))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_profiles(ux, uy, uz, b, x, H; t=nothing, fname="profiles.png")
    H0 = H([x])
    nz = 2*Int64(round(H0/0.01)) + 1
    z = range(-H0, 0, length=nz)
    dz = z[2] - z[1]

    uxs = [nan_eval(ux, Point(x, zᵢ)) for zᵢ ∈ z]
    uys = [nan_eval(uy, Point(x, zᵢ)) for zᵢ ∈ z]
    uzs = [nan_eval(uz, Point(x, zᵢ)) for zᵢ ∈ z]
    bs = [nan_eval(b, Point(x, zᵢ)) for zᵢ ∈ z]
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0
    bzs = (bs[3:end] - bs[1:end-2])/(2dz)
    bzs = [0; bzs; (1/2*bs[end-2] - 2*bs[end-1] + 3/2*bs[end])/dz]

    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    ax[4].set_xlabel(L"\partial_z b")
    # ax[4].set_xlabel(L"b")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    for a ∈ ax 
        a.set_ylim(-H([x]), 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    ax[1].plot(uxs, z)
    ax[2].plot(uys, z)
    ax[3].plot(uzs, z)
    ax[4].plot(bzs, z)
    # ax[4].plot(bs, z)
    if t === nothing
        ax[1].set_title(L"x = "*@sprintf("%1.2f", x))
    else
        ax[1].set_title(L"x = "*@sprintf("%1.2f", x)*L", \quad t = "*@sprintf("%1.2f", t))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_profiles(ux, uy, uz, b, x, y, H; t=nothing, fname="profiles.png")
    H0 = H([x, y])
    nz = 2*Int64(round(H0/0.01)) + 1
    z = range(-H0, 0, length=nz)
    dz = z[2] - z[1]

    uxs = [nan_eval(ux, Point(x, y, zᵢ)) for zᵢ ∈ z]
    uys = [nan_eval(uy, Point(x, y, zᵢ)) for zᵢ ∈ z]
    uzs = [nan_eval(uz, Point(x, y, zᵢ)) for zᵢ ∈ z]
    bs = [nan_eval(b, Point(x, y, zᵢ)) for zᵢ ∈ z]
    bzs = (bs[3:end] - bs[1:end-2])/(2dz)
    bzs = [0; bzs; (1/2*bs[end-2] - 2*bs[end-1] + 3/2*bs[end])/dz]
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0

    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    ax[4].set_xlabel(L"\partial_z b")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    for a ∈ ax 
        a.set_ylim(-H([x, y]), 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    ax[1].plot(uxs, z)
    ax[2].plot(uys, z)
    ax[3].plot(uzs, z)
    ax[4].plot(bzs, z)
    if t === nothing
        ax[1].set_title(L"x = "*@sprintf("%1.2f", x)*L", \quad y = "*@sprintf("%1.2f", y))
    else
        ax[1].set_title(L"x = "*@sprintf("%1.2f", x)*L", \quad y = "*@sprintf("%1.2f", y)*L", \quad t = "*@sprintf("%1.2f", t))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_profiles(b, x, y, H; t=nothing, fname="profiles.png")
    H0 = H([x, y])
    nz = 2*Int64(round(H0/0.01)) + 1
    z = range(-H0, 0, length=nz)
    dz = z[2] - z[1]

    bs = [nan_eval(b, Point(x, y, zᵢ)) for zᵢ ∈ z]
    bzs = (bs[3:end] - bs[1:end-2])/(2dz)
    bzs = [0; bzs; (1/2*bs[end-2] - 2*bs[end-1] + 3/2*bs[end])/dz]

    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"b")
    ax[2].set_xlabel(L"\partial_z b")
    ax[2].set_yticklabels([])
    for a ∈ ax
        a.set_ylim(-H0, 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    ax[1].plot(bs, z)
    ax[2].plot(bzs, z)
    if t === nothing
        ax[1].set_title(L"x = "*@sprintf("%1.2f", x)*L", \quad y = "*@sprintf("%1.2f", y))
    else
        ax[1].set_title(L"x = "*@sprintf("%1.2f", x)*L", \quad y = "*@sprintf("%1.2f", y)*L", \quad t = "*@sprintf("%1.2f", t))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_sparsity_pattern(A; fname="sparsity_pattern.png")
    N = maximum(size(A))
    df = N ÷ 256
    A_downsampled = zeros(size(A, 1) ÷ df, size(A, 2) ÷ df)
    for i ∈ axes(A_downsampled, 1), j ∈ axes(A_downsampled, 2)
        A_downsampled[i, j] = sum(abs.(A[df*i-df+1:df*i, df*j-df+1:df*j]))/df^2
    end
    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    ax.spy(A_downsampled, markersize=0.5, markeredgewidth=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(true)
    ax.spines["right"].set_visible(true)
    plt.savefig(fname)
    println(fname)
    plt.close()
end

function plot_u_sfc(ux, uy, m, m_sfc; t=nothing, fname="u_sfc.png")
    u = unpack_fefunction(ux, m)
    v = unpack_fefunction(uy, m)
    speed = @. sqrt(u^2 + v^2)
    i_sfc = unique(m_sfc.t[:])
    vmax = maximum(abs.(speed[i_sfc]))
    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    img = ax.tripcolor(m_sfc.p[:, 1], m_sfc.p[:, 2], m_sfc.t[:, 1:3] .- 1, speed, shading="gouraud", cmap="Reds", vmin=0, vmax=vmax, rasterized=true)
    n = 1000
    i_show = i_sfc[1:div(length(i_sfc), n):end]
    ax.quiver(m_sfc.p[i_show, 1], m_sfc.p[i_show, 2], u[i_show], v[i_show], color="k")
    plt.colorbar(img, ax=ax, label=L"Surface speed $\sqrt{u^2 + v^2}$")
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    if t === nothing
        ax.set_title("Surface velocity")
    else
        ax.set_title(L"Surface velocity at $t = $"*@sprintf("%1.2f", t))
    end
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    plt.savefig(fname)
    println(fname)
    plt.close()
end