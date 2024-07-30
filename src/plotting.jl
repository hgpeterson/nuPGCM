"""
    quick_plot(u, g; b, label, fname)

Plot a scalar field `u` on a mesh `g`.
"""
function quick_plot(u::AbstractVector, g::MyGrid; b=nothing, label="", fname="image.png")
    fig, ax = plt.subplots(1)
    umax = maximum(abs.(u))
    img = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u, shading="gouraud", vmin=-umax, vmax=umax, cmap="RdBu_r", rasterized=true)
    if b !== nothing
        b = unpack_fefunction(b, g)
        ax.tricontour(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, b, colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    end
    cb = plt.colorbar(img, ax=ax, label=label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig(fname)
    println(fname)
    plt.close()
end
function quick_plot(u::FEFunction, g::MyGrid; kwargs...)
    u = unpack_fefunction(u, g)
    quick_plot(u, g; kwargs...)
end

function plot_yslice(u, b, y, H; t=nothing, cb_label="", fname="yslice.png")
    σ = (chebyshev_nodes(2^6) .- 1)/2
    xs = range(-1, 1, length=2^3)
    Hs = [H([x, y]) for x ∈ xs]

    println("evaling")
    @time us = [nan_eval(u, Point(xs[i], y, Hs[i]*σ[j])) for i ∈ eachindex(xs), j ∈ eachindex(σ)]
    # bs = [nan_eval(b, Point(xs[i], y, Hs[i]*σ[j])) for i ∈ eachindex(xs), j ∈ eachindex(σ)]
    us[:, 1] .= 0

    fig, ax = plt.subplots(1)
    img = ax.pcolormesh(xs, σ, us', shading="gouraud", cmap="RdBu_r", rasterized=true)
    plt.colorbar(img, ax=ax, label=cb_label)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_profiles(ux, uy, uz, b, x, H; t=nothing, fname="profiles.png")
    z = H([x])*(chebyshev_nodes(2^6) .- 1)/2

    uxs = [nan_eval(ux, Point(x, zᵢ)) for zᵢ ∈ z]
    uys = [nan_eval(uy, Point(x, zᵢ)) for zᵢ ∈ z]
    uzs = [nan_eval(uz, Point(x, zᵢ)) for zᵢ ∈ z]
    # bz = VectorValue(0.0, 0.0, 1.0)⋅∇(b)
    # bzs = [nan_eval(bz, Point(x, y, zᵢ)) for zᵢ ∈ z]
    bs = [nan_eval(b, Point(x, zᵢ)) for zᵢ ∈ z]
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0
    # bzs[1] = 0

    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    # ax[4].set_xlabel(L"\partial_z b")
    ax[4].set_xlabel(L"b")
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
    # ax[4].plot(bzs, z)
    ax[4].plot(bs, z)
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
    z = H([x, y])*(chebyshev_nodes(2^6) .- 1)/2

    uxs = [nan_eval(ux, Point(x, y, zᵢ)) for zᵢ ∈ z]
    uys = [nan_eval(uy, Point(x, y, zᵢ)) for zᵢ ∈ z]
    uzs = [nan_eval(uz, Point(x, y, zᵢ)) for zᵢ ∈ z]
    # bz = VectorValue(0.0, 0.0, 1.0)⋅∇(b)
    # bzs = [nan_eval(bz, Point(x, y, zᵢ)) for zᵢ ∈ z]
    bs = [nan_eval(b, Point(x, y, zᵢ)) for zᵢ ∈ z]
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0
    # bzs[1] = 0

    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    # ax[4].set_xlabel(L"\partial_z b")
    ax[4].set_xlabel(L"b")
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
    # ax[4].plot(bzs, z)
    ax[4].plot(bs, z)
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