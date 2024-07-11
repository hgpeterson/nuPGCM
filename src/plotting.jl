"""
    quick_plot(u, g; b, label, fname)

Plot a scalar field `u` on a mesh `g`.
"""
function quick_plot(u, g::MyGrid; b=nothing, label="", fname="image.png")
    u = unpack_fefunction(u, g)
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
    plt.savefig(fname, dpi=400)
    println(fname)
    plt.close()
end