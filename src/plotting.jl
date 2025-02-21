"""
    ux = nan_eval(u::CellField, x::VectorValue)
    ux = nan_eval(u::CellField, x::AbstractVector)
    ux = nan_eval(cache, u::CellField, x::VectorValue)
    ux = nan_eval(cache, u::CellField, x::AbstractVector)

Evaluate `CellField` `u` at point(s) `x` and return `NaN` if an error occurs.
"""
function nan_eval(u::CellField, x::VectorValue)
    cache = Gridap.CellData.return_cache(u, x)
    return nan_eval(cache, u, x)
end
function nan_eval(cache, u::CellField, x::VectorValue)
    try 
        evaluate!(cache, u, x) 
    catch e
        if isa(e, AssertionError) 
            # catch "x is not inside any active cell" error
            return NaN 
        else 
            rethrow(e)
        end
    end
end
function nan_eval(u::CellField, x::AbstractVector)
    cache = Gridap.CellData.return_cache(u, x)
    return nan_eval(cache, u, x)
end
function nan_eval(cache, u::CellField, x::AbstractVector)
    return [nan_eval(cache, u, x[i]) for i ∈ eachindex(x)]
end

"""
    cache = plot_slice(u::CellField, b::CellField, N²::Real,; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", cb_max=0., fname="slice.png")
    plot_slice(cache::Tuple, u::CellField, b::CellField; t=nothing, fname="slice.png")

    cache = plot_slice(u::CellField, v::CellField, b::CellField, N²::Real; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", cb_max=0., fname="slice.png")
    plot_slice(cache::Tuple, u::CellField, v::CellField, b::CellField; t=nothing, fname="slice.png")
"""
function plot_slice(u::CellField, b::CellField, N²::Real; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", cb_max=0., fname="slice.png")
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
    cache = (slice_dir, x, y, z, N², points, cache_u, cache_b, cb_label, cb_max)

    # plot
    plot_slice(cache, u, b; t, fname)
    
    # return cache for future use
    return cache
end
function plot_slice(cache::Tuple, u::CellField, b::CellField; t=nothing, fname="slice.png")
    # unpack cache
    slice_dir, x, y, z, N², points, cache_u, cache_b, cb_label, cb_max = cache

    # evaluate
    us = [nan_eval(cache_u, u, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    bs = [N²*points[i, j][3] + nan_eval(cache_b, b, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]

    # fill NaNs
    nx = size(points, 1)
    nz = size(points, 2)
    for i ∈ 1:nx, j ∈ 1:nz 
        if isnan(bs[i, j])
            if !isnan(bs[max(i-1, 1), j]) && !isnan(bs[min(i+1, nx), j]) && !isnan(bs[i, max(j-1, 1)]) && !isnan(bs[i, min(j+1, nz)])
                bs[i, j] = (bs[max(i-1, 1), j] + bs[min(i+1, nx), j] + bs[i, max(j-1, 1)] + bs[i, min(j+1, nz)])/4
            end
        end
        if isnan(us[i, j])
            if !isnan(us[max(i-1, 1), j]) && !isnan(us[min(i+1, nx), j]) && !isnan(us[i, max(j-1, 1)]) && !isnan(us[i, min(j+1, nz)])
                us[i, j] = (us[max(i-1, 1), j] + us[min(i+1, nx), j] + us[i, max(j-1, 1)] + us[i, min(j+1, nz)])/4
            end
        end
    end

    # plot
    fig, ax = plt.subplots(1)
    ax.axis("equal")
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
    if cb_max == 0.
        cb_max = nan_max(abs.(us))
    end
    img = ax.pcolormesh(x1, x2, us', shading="nearest", cmap="RdBu_r", vmin=-cb_max, vmax=cb_max, rasterized=true) # need to use nearest for NaNs in us
    cb = plt.colorbar(img, ax=ax, label=cb_label, fraction=0.025)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x1, x2, bs', colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    if t === nothing
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$", slice_dir, slice_coord)))
    else
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$, \$t = %s\$", slice_dir, slice_coord, sci_notation(t))))
    end
    savefig(fname)
    println(fname)
    plt.close()
end
function plot_slice(u::CellField, v::CellField, b::CellField, N²::Real; x=nothing, y=nothing, z=nothing, t=nothing, cb_label="", cb_max=0., fname="slice.png")
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
    cache = (slice_dir, x, y, z, N², points, cache_u, cache_v, cache_b, cb_label, cb_max)

    # plot
    plot_slice(cache, u, v, b; t, fname)
    
    # return cache for future use
    return cache
end
function plot_slice(cache::Tuple, u::CellField, v::CellField, b::CellField; t=nothing, fname="slice.png")
    # unpack cache
    slice_dir, x, y, z, N², points, cache_u, cache_v, cache_b, cb_label, cb_max = cache

    # evaluate
    us = [nan_eval(cache_u, u, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    vs = [nan_eval(cache_v, v, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    bs = [N²*points[i, j][3] + nan_eval(cache_b, b, points[i, j]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    speed = @. sqrt(us^2 + vs^2)

    # fill NaNs
    nx = size(points, 1)
    nz = size(points, 2)
    for i ∈ 1:nx, j ∈ 1:nz 
        if isnan(bs[i, j])
            if !isnan(bs[max(i-1, 1), j]) && !isnan(bs[min(i+1, nx), j]) && !isnan(bs[i, max(j-1, 1)]) && !isnan(bs[i, min(j+1, nz)])
                bs[i, j] = (bs[max(i-1, 1), j] + bs[min(i+1, nx), j] + bs[i, max(j-1, 1)] + bs[i, min(j+1, nz)])/4
            end
        end
        if isnan(us[i, j])
            if !isnan(us[max(i-1, 1), j]) && !isnan(us[min(i+1, nx), j]) && !isnan(us[i, max(j-1, 1)]) && !isnan(us[i, min(j+1, nz)])
                us[i, j] = (us[max(i-1, 1), j] + us[min(i+1, nx), j] + us[i, max(j-1, 1)] + us[i, min(j+1, nz)])/4
            end
        end
    end

    # plot
    fig, ax = plt.subplots(1)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
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
    if cb_max == 0.
        cb_max = nan_max(speed)
        scale = nothing
        scale_units = nothing
    else
        scale = cb_max/0.1 # speed of `cb_max` corresponds to 0.1 inch arrow
        scale_units = "inches"
    end
    img = ax.pcolormesh(x1, x2, speed', shading="nearest", cmap="Reds", vmin=0, vmax=cb_max, rasterized=true) # need to use nearest for NaNs in us
    cb = plt.colorbar(img, ax=ax, label=cb_label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x1, x2, bs', colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    n = length(x1)
    slice = 1:2^3:n
    ax.quiver(x1[slice], x2[slice], us[slice, slice]', vs[slice, slice]', color="k", pivot="mid", scale=scale, scale_units=scale_units)
    if t === nothing
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$", slice_dir, slice_coord)))
    else
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$, \$t = %s\$", slice_dir, slice_coord, sci_notation(t))))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

"""
    cache = plot_profiles(ux::CellField, uy::CellField, uz::CellField, b::CellField, N²::Real, H::Function; 
                          x::Real, y::Real, t=nothing, fname="profiles.png")
    plot_profiles(cache::Tuple, ux::CellField, uy::CellField, uz::CellField, b::CellField
                  t=nothing, fname="profiles.png") 
"""
function plot_profiles(ux::CellField, uy::CellField, uz::CellField, b::CellField, N²::Real,
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
    cache = (x, y, z, N², points, cache_ux, cache_uy, cache_uz, cache_b)

    # plot
    plot_profiles(cache, ux, uy, uz, b; t, fname)

    # return cache for future use
    return cache
end
function plot_profiles(cache::Tuple, ux::CellField, uy::CellField, uz::CellField, b::CellField;
                       t=nothing, fname="profiles.png")
    # unpack cache
    x, y, z, N², points, cache_ux, cache_uy, cache_uz, cache_b = cache

    # evaluate fields
    uxs = nan_eval(cache_ux, ux, points)
    uys = nan_eval(cache_uy, uy, points)
    uzs = nan_eval(cache_uz, uz, points)
    bs  = nan_eval(cache_b,  b,  points)

    # compute bz
    dz = z[2] - z[1]
    bzs = (bs[3:end] - bs[1:end-2])/(2dz)
    bzs = [(-3/2*bs[1] + 2*bs[2] - 1/2*bs[3])/dz; bzs; (1/2*bs[end-2] - 2*bs[end-1] + 3/2*bs[end])/dz]
    bzs .+= N²
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0

    # NaN masks
    ux_mask = (isnan.(uxs) .== 0)
    uy_mask = (isnan.(uys) .== 0)
    uz_mask = (isnan.(uzs) .== 0)
    bz_mask = (isnan.(bzs) .== 0)

    # # print integrals
    # @printf("∫ u dz = %e\n", trapz(uxs[ux_mask], z[ux_mask]))
    # @printf("∫ v dz = %e\n", trapz(uys[uy_mask], z[uy_mask]))
    # @printf("∫ w dz = %e\n", trapz(uzs[uz_mask], z[uz_mask]))

    # plot
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
    for a ∈ ax[1:3]
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", linewidth=0.5, linestyle="-")
    end
    ax[4].set_xlim(0, 1.1*nan_max(abs.(bzs)))
    ax[1].plot(uxs[ux_mask], z[ux_mask])
    ax[2].plot(uys[uy_mask], z[uy_mask])
    ax[3].plot(uzs[uz_mask], z[uz_mask])
    ax[4].plot(bzs[bz_mask], z[bz_mask])
    if t === nothing
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f", x, y)))
    else
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f, \\quad t = %s", x, y, sci_notation(t))))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

"""
    cache = sim_plots(dim::AbstractDimension, ux::CellField, uy::CellField, uz::CellField, b::CellField, N²::Real, H::Function, t::Real, i_save::Int)
    sim_plots(cache::Tuple, ux::CellField, uy::CellField, uz::CellField, b::CellField, t::Real, i_save::Int)
"""
function sim_plots(dim::TwoD, ux::CellField, uy::CellField, uz::CellField, b::CellField, N²::Real, H::Function, t::Real, i_save::Int)
    # don't plot u_sfc in 2D
    @time "plotting" begin
    cache_profiles = plot_profiles(ux, uy, uz, b, N², H; x=0.5, y=0.0, t=t, fname=@sprintf("%s/images/profiles%03d.png", out_dir, i_save))
    cache_u_slice  = plot_slice(ux, b, N²; y=0.0, t=t, cb_label=L"Zonal flow $u$",      fname=@sprintf("%s/images/u_yslice_%03d.png", out_dir, i_save))
    cache_v_slice  = plot_slice(uy, b, N²; y=0.0, t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_yslice_%03d.png", out_dir, i_save))
    cache_w_slice  = plot_slice(uz, b, N²; y=0.0, t=t, cb_label=L"Vertical flow $w$",   fname=@sprintf("%s/images/w_yslice_%03d.png", out_dir, i_save))
    end
    return cache_profiles, cache_u_slice, cache_v_slice, cache_w_slice
end
function sim_plots(dim::ThreeD, ux::CellField, uy::CellField, uz::CellField, b::CellField, N²::Real, H::Function, t::Real, i_save::Int)
    @time "plotting" begin
    cache_profiles = plot_profiles(ux, uy, uz, b, N², H; x=0.5, y=0.0, t=t, fname=@sprintf("%s/images/profiles%03d.png", out_dir, i_save))
    cache_u_slice  = plot_slice(ux, b, N²; y=0.0, t=t, cb_label=L"Zonal flow $u$",      fname=@sprintf("%s/images/u_yslice_%03d.png", out_dir, i_save))
    cache_v_slice  = plot_slice(uy, b, N²; y=0.0, t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_yslice_%03d.png", out_dir, i_save))
    cache_w_slice  = plot_slice(uz, b, N²; y=0.0, t=t, cb_label=L"Vertical flow $w$",   fname=@sprintf("%s/images/w_yslice_%03d.png", out_dir, i_save))
    cache_u_sfc    = plot_slice(ux, uy, b, N²; z=0.0, t=t, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", fname=@sprintf("%s/images/u_sfc_%03d.png", out_dir, i_save))
    end
    return cache_profiles, cache_u_slice, cache_v_slice, cache_w_slice, cache_u_sfc
end
function sim_plots(cache::Tuple, ux::CellField, uy::CellField, uz::CellField, b::CellField, t::Real, i_save::Int)
    if length(cache) == 5
        # 3D
        cache_profiles, cache_u_slice, cache_v_slice, cache_w_slice, cache_u_sfc = cache
        plot_u_sfc = true
    else
        # 2D
        cache_profiles, cache_u_slice, cache_v_slice, cache_w_slice = cache
        plot_u_sfc = false
    end

    @time "plotting" begin
    plot_profiles(cache_profiles, ux, uy, uz, b; t=t, fname=@sprintf("%s/images/profiles%03d.png", out_dir, i_save))
    plot_slice(cache_u_slice, ux, b; t=t, fname=@sprintf("%s/images/u_yslice_%03d.png", out_dir, i_save))
    plot_slice(cache_v_slice, uy, b; t=t, fname=@sprintf("%s/images/v_yslice_%03d.png", out_dir, i_save))
    plot_slice(cache_w_slice, uz, b; t=t, fname=@sprintf("%s/images/w_yslice_%03d.png", out_dir, i_save))
    if plot_u_sfc
        plot_slice(cache_u_sfc,   ux, uy, b; t=t, fname=@sprintf("%s/images/u_sfc_%03d.png", out_dir, i_save))
    end
    end
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

# for eigen.jl
function plot_slice_wave(u::CellField, b::CellField, N²::Real, k::Real, ω::Complex; 
                         x=nothing, y=nothing, z=nothing, cb_label="", cb_max=0., fname="slice.png", 
                         fig=nothing, ax=nothing)
    # setup grid and cache
    n = 2^8
    if x !== nothing
        y = range(-1, 1, length=n)
        z = range(-1, 0, length=n)
        points = [Point(x, yᵢ, zᵢ) for yᵢ ∈ y, zᵢ ∈ z]
        slice_dir = "x"
    elseif y !== nothing
        x = range(-pi/k, pi/k, length=n)
        z = range(-1, 0, length=n)
        points = [Point(xᵢ, y, zᵢ) for xᵢ ∈ x, zᵢ ∈ z]
        slice_dir = "y"
    elseif z !== nothing
        x = range(-pi/k, pi/k, length=n)
        y = range(-1, 1, length=n)
        points = [Point(xᵢ, yᵢ, z) for xᵢ ∈ x, yᵢ ∈ y]
        slice_dir = "z"
    else
        error("One of x, y, or z must be specified for slice.")
    end
    cache_u = Gridap.CellData.return_cache(u, points[:])
    cache_b = Gridap.CellData.return_cache(b, points[:])
    cache = (slice_dir, x, y, z, N², k, ω, points, cache_u, cache_b, cb_label, cb_max)

    # plot
    plot_slice_wave(cache, u, b; fname, fig, ax)
    
    # return cache for future use
    return cache
end
function plot_slice_wave(cache::Tuple, u::CellField, b::CellField; fname="slice.png", fig=nothing, ax=nothing)
    # unpack cache
    slice_dir, x, y, z, N², k, ω, points, cache_u, cache_b, cb_label, cb_max = cache

    # evaluate
    us = [nan_eval(cache_u, u, points[i, j])*exp(im*k*points[i, j][1]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]
    bs = [N²*points[i, j][3] + nan_eval(cache_b, b, points[i, j])*exp(im*k*points[i, j][1]) for i ∈ axes(points, 1), j ∈ axes(points, 2)]

    # fill NaNs
    nx = size(points, 1)
    nz = size(points, 2)
    for i ∈ 1:nx, j ∈ 1:nz 
        if isnan(bs[i, j])
            if !isnan(bs[max(i-1, 1), j]) && !isnan(bs[min(i+1, nx), j]) && !isnan(bs[i, max(j-1, 1)]) && !isnan(bs[i, min(j+1, nz)])
                bs[i, j] = (bs[max(i-1, 1), j] + bs[min(i+1, nx), j] + bs[i, max(j-1, 1)] + bs[i, min(j+1, nz)])/4
            end
        end
        if isnan(us[i, j])
            if !isnan(us[max(i-1, 1), j]) && !isnan(us[min(i+1, nx), j]) && !isnan(us[i, max(j-1, 1)]) && !isnan(us[i, min(j+1, nz)])
                us[i, j] = (us[max(i-1, 1), j] + us[min(i+1, nx), j] + us[i, max(j-1, 1)] + us[i, min(j+1, nz)])/4
            end
        end
    end

    # real part
    us = real(us)
    bs = real(bs)

    # plot
    if fig === nothing && ax === nothing
        fig, ax = plt.subplots(1)
        save_plot = true
    else
        save_plot = false
    end
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
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
        ax.set_xticks([-pi/k, 0, pi/k])
        ax.set_xticklabels([L"-\pi/k", L"0", L"\pi/k"])
        ax.set_yticks(-1:0.5:0)
    elseif slice_dir == "z"
        slice_coord = z
        x1 = x
        x2 = y
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        ax.set_xticks([-pi/k, 0, pi/k])
        ax.set_xticklabels([L"-\pi/k", L"0", L"\pi/k"])
        ax.set_yticks(-0.5:0.5:0.5)
    end
    ax.axis("equal")
    if cb_max == 0.
        cb_max = nan_max(abs.(us))
    end
    img = ax.pcolormesh(x1, x2, us', shading="nearest", cmap="RdBu_r", vmin=-cb_max, vmax=cb_max, rasterized=true) # need to use nearest for NaNs in us
    if save_plot
        cb = plt.colorbar(img, ax=ax, label=cb_label, fraction=0.025)
        cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
        ax.set_title(latexstring(@sprintf("Slice at \$%s = %1.2f\$", slice_dir, slice_coord)))
        if imag(ω) >= 0
            ax.annotate(latexstring(@sprintf("\$\\omega = %0.2f + %0.2f i\$", real(ω), +imag(ω))), xy=(0.05, 0.9), xycoords="axes fraction")
        else
            ax.annotate(latexstring(@sprintf("\$\\omega = %0.2f - %0.2f i\$", real(ω), -imag(ω))), xy=(0.05, 0.9), xycoords="axes fraction")
        end
    end
    # ax.contour(x1, x2, bs', colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    if save_plot
        savefig(fname)
        println(fname)
        plt.close()
    end
end