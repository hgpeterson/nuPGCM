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

"""
    cache = plot_slice(u::CellField, b::CellField; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", fname="slice.png")
    cache = plot_slice(u::CellField, v::CellField, b::CellField; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", fname="slice.png")

    plot_slice(cache::Tuple, u::CellField, b::CellField; t=nothing, cb_label="", fname="slice.png")
    plot_slice(cache::Tuple, u::CellField, v::CellField, b::CellField; t=nothing, cb_label="", fname="slice.png")
"""
function plot_slice(u::CellField, b::CellField; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", fname="slice.png")
    # setup grid and cache
    n = 2^8
    if x !== nothing
        y = range(-1, 1, length=n)
        z = range(-1, 0, length=n)
        points = [Point(x, yᵢ, zᵢ) for yᵢ ∈ y, zᵢ ∈ z]
        slice_dir = "x"
    elseif y !== nothing
        x = range(-1, 1, length=n)
        z = range(-1, 0, length=n)
        points = [Point(xᵢ, y, zᵢ) for xᵢ ∈ x, zᵢ ∈ z]
        slice_dir = "y"
    elseif z !== nothing
        x = range(-1, 1, length=n)
        y = range(-1, 1, length=n)
        points = [Point(xᵢ, yᵢ, z) for xᵢ ∈ x, yᵢ ∈ y]
        slice_dir = "z"
    else
        error("One of x, y, or z must be specified for slice.")
    end
    cache_u = Gridap.CellData.return_cache(u, points[:])
    cache_b = Gridap.CellData.return_cache(b, points[:])
    cache = (slice_dir, x, y, z, points, cache_u, cache_b)

    # plot
    plot_slice(cache, u, b; t, cb_label, fname)
    
    # return cache for future use
    return cache
end
function plot_slice(u::CellField, v::CellField, b::CellField; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", fname="slice.png")
    # setup grid and cache
    n = 2^8
    if x !== nothing
        y = range(-1, 1, length=n)
        z = range(-1, 0, length=n)
        points = [Point(x, yᵢ, zᵢ) for yᵢ ∈ y, zᵢ ∈ z]
        slice_dir = "x"
    elseif y !== nothing
        x = range(-1, 1, length=n)
        z = range(-1, 0, length=n)
        points = [Point(xᵢ, y, zᵢ) for xᵢ ∈ x, zᵢ ∈ z]
        slice_dir = "y"
    elseif z !== nothing
        x = range(-1, 1, length=n)
        y = range(-1, 1, length=n)
        points = [Point(xᵢ, yᵢ, z) for xᵢ ∈ x, yᵢ ∈ y]
        slice_dir = "z"
    else
        error("One of x, y, or z must be specified for slice.")
    end
    cache_u = Gridap.CellData.return_cache(u, points[:])
    cache_v = Gridap.CellData.return_cache(v, points[:])
    cache_b = Gridap.CellData.return_cache(b, points[:])
    cache = (slice_dir, x, y, z, points, cache_u, cache_v, cache_b)

    # plot
    plot_slice(cache, u, v, b; t, cb_label, fname)
    
    # return cache for future use
    return cache
end
function plot_slice(cache::Tuple, u::CellField, b::CellField; t=nothing, cb_label="", fname="slice.png")
    # unpack cache
    slice_dir, x, y, z, points, cache_u, cache_b = cache

    # evaluate
    us = [nan_eval(cache_u, u, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    bs = [nan_eval(cache_b, b, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    # us[isnan.(us)] .= 0

    # plot
    fig, ax = plt.subplots(1)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.axis("equal")
    if slice_dir == "x"
        slice_coord = x
        x1 = y 
        x2 = z
        ax.set_xlabel(L"y")
        ax.set_ylabel(L"z")
        ax.set_xticks(-1:0.5:1)
        ax.set_yticks(-1:0.5:0)
    elseif slice_dir == "y"
        slice_coord = y
        x1 = x
        x2 = z
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"z")
        ax.set_xticks(-1:0.5:1)
        ax.set_yticks(-1:0.5:0)
    elseif slice_dir == "z"
        slice_coord = z
        x1 = x
        x2 = y
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        ax.set_xticks(-1:0.5:1)
        ax.set_yticks(-1:0.5:1)
    end
    umax = nan_max(abs.(us))
    img = ax.pcolormesh(x1, x2, us', shading="gouraud", cmap="RdBu_r", vmin=-umax, vmax=umax, rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=cb_label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x1, x2, bs', colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    if t === nothing
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$", slice_dir, slice_coord)))
    else
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$, \$t = %1.2f\$", slice_dir, slice_coord, t)))
    end
    savefig(fname)
    println(fname)
    plt.close()
end
function plot_slice(cache::Tuple, u::CellField, v::CellField, b::CellField; t=nothing, cb_label="", fname="slice.png")
    # unpack cache
    slice_dir, x, y, z, points, cache_u, cache_v, cache_b = cache

    # evaluate
    us = [nan_eval(cache_u, u, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    vs = [nan_eval(cache_v, v, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    bs = [nan_eval(cache_b, b, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    speed = @. sqrt(us^2 + vs^2)

    # plot
    fig, ax = plt.subplots(1)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.axis("equal")
    if slice_dir == "x"
        slice_coord = x
        x1 = y 
        x2 = z
        ax.set_xlabel(L"y")
        ax.set_ylabel(L"z")
        ax.set_xticks(-1:0.5:1)
        ax.set_yticks(-1:0.5:0)
    elseif slice_dir == "y"
        slice_coord = y
        x1 = x
        x2 = z
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"z")
        ax.set_xticks(-1:0.5:1)
        ax.set_yticks(-1:0.5:0)
    elseif slice_dir == "z"
        slice_coord = z
        x1 = x
        x2 = y
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        ax.set_xticks(-1:0.5:1)
        ax.set_yticks(-1:0.5:1)
    end
    img = ax.pcolormesh(x1, x2, speed', shading="gouraud", cmap="Reds", vmin=0, vmax=maximum(speed), rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=cb_label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x1, x2, bs', colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    n = length(x1)
    slice = 1:2^4:n
    ax.quiver(x1[slice], x2[slice], us[slice, slice], vs[slice, slice], color="k")
    if t === nothing
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$", slice_dir, slice_coord)))
    else
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$, \$t = %1.2f\$", slice_dir, slice_coord, t)))
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

"""
    cache = plot_profiles(ux::CellField, uy::CellField, uz::CellField, b::CellField, H::Function; 
                          x::Real, y::Real, t=nothing, fname="profiles.png")
    plot_profiles(cache::Tuple, ux::CellField, uy::CellField, uz::CellField, b::CellField
                  t=nothing, fname="profiles.png") 
"""
function plot_profiles(ux::CellField, uy::CellField, uz::CellField, b::CellField, 
                       H::Function; x::Real, y::Real, t=nothing, fname="profiles.png")
    # setup points
    H0 = H([x, y])
    z = range(-H0, 0, length=2^8)
    points = [Point(x, y, zᵢ) for zᵢ ∈ z]

    # compute evaluation caches
    cache_ux = Gridap.CellData.return_cache(ux, points)
    cache_uy = Gridap.CellData.return_cache(uy, points)
    cache_uz = Gridap.CellData.return_cache(uz, points)
    cache_b  = Gridap.CellData.return_cache(b,  points)

    # save plotting cache
    cache = (x, y, z, points, cache_ux, cache_uy, cache_uz, cache_b)

    # plot
    plot_profiles(cache, ux, uy, uz, b; t, fname)

    # return cache for future use
    return cache
end
function plot_profiles(cache::Tuple, ux::CellField, uy::CellField, uz::CellField, b::CellField;
                       t=nothing, fname="profiles.png")
    # unpack cache
    x, y, z, points, cache_ux, cache_uy, cache_uz, cache_b = cache

    # evaluate fields
    uxs = nan_eval(cache_ux, ux, points)
    uys = nan_eval(cache_uy, uy, points)
    uzs = nan_eval(cache_uz, uz, points)
    bs  = nan_eval(cache_b,  b,  points)

    # compute bz
    dz = z[2] - z[1]
    bzs = (bs[3:end] - bs[1:end-2])/(2dz)
    bzs = [(-3/2*bs[1] + 2*bs[2] - 1/2*bs[3])/dz; bzs; (1/2*bs[end-2] - 2*bs[end-1] + 3/2*bs[end])/dz]
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
        a.set_ylim(z[1], 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    ax[4].set_xlim(0, 1.1*nan_max(abs.(bzs)))
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

"""
    ux = nan_eval(u::CellField, x::VectorValue)
    ux = nan_eval(u::CellField, x::AbstractVector)
    ux = nan_eval(cache, u::CellField, x::VectorValue)
    ux = nan_eval(cache, u::CellField, x::AbstractVector)

Evaluate `CellField` `u` at point(s) `x` and return `NaN` if an error occurs.
"""
function nan_eval(u::CellField, x::VectorValue)
    try 
        evaluate(u, x) 
    catch 
        NaN 
    end
end
function nan_eval(cache, u::CellField, x::VectorValue)
    try 
        evaluate!(cache, u, x) 
    catch 
        NaN 
    end
end
function nan_eval(u::CellField, x::AbstractVector)
    return [nan_eval(u, xᵢ) for xᵢ ∈ x]
end
function nan_eval(cache, u::CellField, x::AbstractVector)
    return [nan_eval(cache, u, x[i]) for i ∈ eachindex(x)]
end

"""
    u = unpack_fefunction(u, m::Mesh)

Unpack Gridap finite element function `u` into a vector of values at the nodes 
of the mesh. (Assumes `u` is continuous).
"""
function unpack_fefunction(u, m::Mesh)
    u_cell_values = get_cell_dof_values(u)
    return [u_cell_values[m.p_to_t[i][1][1]][m.p_to_t[i][1][2]] for i ∈ 1:size(m.p, 1)]

    # this works for order 1 spaces
    # return sortslices([U.space.metadata.free_dof_to_node       u.free_values
    #                    U.space.metadata.dirichlet_dof_to_node  U.dirichlet_values], dims=1)[:, 2]
end