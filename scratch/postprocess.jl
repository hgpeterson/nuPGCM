using nuPGCM
using Gridap, GridapGmsh
using Printf
using PyPlot
using JLD2

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

# # simulation folder
# out_folder = "../sims/sim035"

# # dimensions
# dim = ThreeD()

# # model
# hres = 0.01
# model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_%0.2f.msh", dim, hres))

# # FE spaces
# X, Y, B, D = setup_FESpaces(model)
# Ux, Uy, Uz, P = unpack_spaces(X)

# # triangulation
# Ω = Triangulation(model)

# # depth
# H(x) = 1 - x[1]^2 - x[2]^2

# # stratification
# N² = 1

# # load state file
# i_save = 3
# statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
# ux, uy, uz, p, b, t = load_state(statefile)
# ux = FEFunction(Ux, ux)
# uy = FEFunction(Uy, uy)
# uz = FEFunction(Uz, uz)
# p  = FEFunction(P, p)
# b  = FEFunction(B, b)

# plot_profiles(ux, uy, uz, b, 1, H; x=0.5, y=0.0, t=t, fname="profiles.png")

# # save vtu
# save_state_vtu(ux, uy, uz, p, b, Ω; fname=@sprintf("%s/data/state%03d.vtu", out_folder, i_save))

# plot_slice(ux, b; x=0,    t=t, cb_label=L"Zonal flow $u$", fname=@sprintf("%s/images/u_xslice_%03d.png", out_folder, i_save))
# plot_slice(ux, b; y=0,    t=t, cb_label=L"Zonal flow $u$", fname=@sprintf("%s/images/u_yslice_%03d.png", out_folder, i_save))
# plot_slice(ux, b; z=-0.5, t=t, cb_label=L"Zonal flow $u$", fname=@sprintf("%s/images/u_zslice_%03d.png", out_folder, i_save))
# plot_slice(uy, b; x=0,    t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_xslice_%03d.png", out_folder, i_save))
# plot_slice(uy, b; y=0,    t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_yslice_%03d.png", out_folder, i_save))
# plot_slice(uy, b; z=-0.5, t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_zslice_%03d.png", out_folder, i_save))
# plot_slice(uz, b; x=0,    t=t, cb_label=L"Vertical flow $w$", fname=@sprintf("%s/images/w_xslice_%03d.png", out_folder, i_save))
# plot_slice(uz, b; y=0,    t=t, cb_label=L"Vertical flow $w$", fname=@sprintf("%s/images/w_yslice_%03d.png", out_folder, i_save))
# plot_slice(uz, b; z=-0.5, t=t, cb_label=L"Vertical flow $w$", fname=@sprintf("%s/images/w_zslice_%03d.png", out_folder, i_save))
# plot_slice(ux, uy, b; z=0.0, t=t, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", fname=@sprintf("%s/images/uv_zslice_%03d.png", out_folder, i_save))
# plot_slice(ux, uz, b; y=0.0, t=t, cb_label=L"Speed $\sqrt{u^2 + w^2}$", fname=@sprintf("%s/images/uw_yslice_%03d.png", out_folder, i_save))
# println()

function plot_animation()
    # for i_save ∈ 1:50
    for i_save ∈ [10]
        statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
        ux, uy, uz, p, b, t = load_state(statefile)
        ux = FEFunction(Ux, ux)
        uy = FEFunction(Uy, uy)
        x, y, U, V, mask = compute_barotropic_velocities(ux, uy)
        Psi = compute_barotropic_streamfunction(U, y)
        plot_barotropic(x, y, U.*mask, V.*mask, Psi.*mask; i=i_save, t=t)
    end
end

function extract_profile_data(x, y; i_save=0)
    # load state file
    statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
    ux, uy, uz, p, b, t = load_state(statefile)
    ux = FEFunction(Ux, ux)
    uy = FEFunction(Uy, uy)
    uz = FEFunction(Uz, uz)
    p  = FEFunction(P, p)
    b  = FEFunction(B, b)

    # setup points
    H0 = H([x, y])
    z = range(-H0, 0, length=2^8)
    points = [Point(x, y, zᵢ) for zᵢ ∈ z]

    # evaluate fields
    uxs = nan_eval(ux, points)
    uys = nan_eval(uy, points)
    uzs = nan_eval(uz, points)
    ps  = nan_eval(p, points)
    bs  = nan_eval(b,  points)

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
    p_mask  = (isnan.(ps)  .== 0)
    bz_mask = (isnan.(bzs) .== 0)

    # save data
    jldsave(@sprintf("%s/data/profiles%03d.jld2", out_folder, i_save); x, y, z, t, uxs, uys, uzs, ps, bs, bzs, ux_mask, uy_mask, uz_mask, p_mask, bz_mask)
    println(@sprintf("%s/data/profiles%03d.jld2", out_folder, i_save))
end

function plot_profiles(datafiles; labels=nothing, fname="profiles.png")
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
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    for a ∈ ax[1:3]
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", linewidth=0.5, linestyle="-")
    end
    ax[4].set_xlim(0, 1.5)
    x = 0.
    y = 0.
    z = nothing
    t = nothing
    jldopen("data/1D.jld2") do file
        z = file["z"]
        u = file["u"]
        v = file["v"]
        w = file["w"]
        b = file["b"]
        bz = differentiate(b, z)
        ax[1].plot(u,       z, "k--", lw=0.5, label="1D", zorder=length(datafiles)+1)
        ax[2].plot(v,       z, "k--", lw=0.5, label="1D", zorder=length(datafiles)+1)
        ax[3].plot(w,       z, "k--", lw=0.5, label="1D", zorder=length(datafiles)+1)
        ax[4].plot(1 .+ bz, z, "k--", lw=0.5, label="1D", zorder=length(datafiles)+1)
    end
    for i ∈ eachindex(datafiles)
        d = datafiles[i]
        jldopen(d) do file
            x = file["x"]
            y = file["y"]
            z = file["z"]
            uxs = file["uxs"]
            uys = file["uys"]
            uzs = file["uzs"]
            bzs = file["bzs"]
            ux_mask = file["ux_mask"]
            uy_mask = file["uy_mask"]
            uz_mask = file["uz_mask"]
            bz_mask = file["bz_mask"]
            label = labels === nothing ? d : labels[i]
            ax[1].plot(uxs[ux_mask], z[ux_mask], label=label)
            ax[2].plot(uys[uy_mask], z[uy_mask])
            ax[3].plot(uzs[uz_mask], z[uz_mask])
            ax[4].plot(bzs[bz_mask], z[bz_mask])
        end
    end
    for a ∈ ax
        a.set_ylim(z[1], 0) 
    end
    ax[1].legend()
    if t === nothing
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f", x, y)))
    else
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f, \\quad t = %s", x, y, sci_notation(t))))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function compute_barotropic_velocities(ux, uy)
    # resolution
    nx = 2^7
    ny = nx
    nz = 2^5

    # 3D grid
    x = range(-1, 1, length=nx)
    y = range(-1, 1, length=ny)
    # z = range(-1, 0, length=nz)
    σ = (chebyshev_nodes(nz) .- 1)/2

    # # horizontal grid
    # p, t = get_p_t("../meshes/circle.msh")
    # x = p[:, 1]
    # y = p[:, 2]
    # nx = ny = size(x, 1)
    # println("nx = ny = ", nx)
    # t = t .- 1

    # points 
    # points = [Point(x[i], y[j], z[k]) for i ∈ 1:nx, j ∈ 1:ny, k ∈ 1:nz][:]
    points = [Point(x[i], y[j], H([x[i], y[j]])*σ[k]) for i ∈ 1:nx, j ∈ 1:ny, k ∈ 1:nz][:]

    # evaluation caches
    cache_ux = Gridap.CellData.return_cache(ux, points)
    cache_uy = Gridap.CellData.return_cache(uy, points)

    # evaluate fields
    @time "evals" begin
    uxs = reshape(nan_eval(cache_ux, ux, points), nx, ny, nz)
    uys = reshape(nan_eval(cache_uy, uy, points), nx, ny, nz)
    end

    # barotropic velocities
    Ux = [trapz(uxs[i, j, :], σ*H([x[i], y[j]])) for i ∈ 1:nx, j ∈ 1:ny]
    Uy = [trapz(uys[i, j, :], σ*H([x[i], y[j]])) for i ∈ 1:nx, j ∈ 1:ny]
    # Ux = [trapz(uxs[i, j, :], z) for i ∈ 1:nx, j ∈ 1:ny]
    # Uy = [trapz(uys[i, j, :], z) for i ∈ 1:nx, j ∈ 1:ny]

    # NaNs outside of domain
    mask = ones(nx, ny)
    for i ∈ 1:nx, j ∈ 1:ny
        if all(isnan.(uxs[i, j, :]))
            mask[i, j] = NaN
        end
    end

    return x, y, Ux, Uy, mask
end

function compute_barotropic_streamfunction(Ux, y)
    Psi = zeros(size(Ux))
    for i in axes(Ux, 1)
        Psi[i, :] = -cumtrapz(Ux[i, :], y)
    end
    return Psi
end

function plot_barotropic(x, y, Ux, Uy, Psi; i=0, t=nothing)
    if t === nothing
        title = ""
    else
        title = latexstring(@sprintf("t = %s", sci_notation(t)))
    end

    get_levels(vmax) = [-vmax, -3vmax/4, -vmax/2, -vmax/4, vmax/4, vmax/2, 3vmax/4, vmax]

    # plot UV
    if i == 0
        fname = @sprintf("%s/images/UV.png", out_folder)
    else
        fname = @sprintf("%s/images/UV%03d.png", out_folder, i)
    end
    fname = "UV.png"
    fig, ax = plt.subplots(1, 2, figsize=(3.2*2, 2))
    ax[1].set_xlabel(L"x")
    ax[1].set_ylabel(L"y")
    ax[2].set_xlabel(L"x")
    ax[2].set_ylabel(L"y")
    ax[1].axis("equal")
    ax[2].axis("equal")
    ax[1].spines["left"].set_visible(false)
    ax[1].spines["bottom"].set_visible(false)
    ax[2].spines["left"].set_visible(false)
    ax[2].spines["bottom"].set_visible(false)
    ax[1].set_xticks(-1:1:1)
    ax[1].set_yticks(-1:1:1)
    ax[2].set_xticks(-1:1:1)
    ax[2].set_yticks(-1:1:1)
    ax[1].set_title(title)
    ax[2].set_title(title)

    vmax = nan_max(abs.(Ux))
    img = ax[1].pcolormesh(x, y, Ux', shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true) 
    cb = plt.colorbar(img, ax=ax[1], label=L"U")
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax[1].contour(x, y, Ux', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(vmax))

    vmax = nan_max(abs.(Uy))
    img = ax[2].pcolormesh(x, y, Uy', shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true) 
    cb = plt.colorbar(img, ax=ax[2], label=L"V")
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax[2].contour(x, y, Uy', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(vmax))
    subplots_adjust(wspace=0.4)
    savefig(fname)
    println(fname)
    plt.close()

    # plot Psi
    if i == 0
        fname = @sprintf("%s/images/psi.png", out_folder)
    else
        fname = @sprintf("%s/images/psi%03d.png", out_folder, i)
    end
    fname = "psi.png"
    fig, ax = plt.subplots(1)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks(-1:1:1)
    ax.set_yticks(-1:1:1)
    ax.set_title(title)
    vmax = nan_max(abs.(Psi))
    img = ax.pcolormesh(x, y, Psi', shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=L"\Psi")
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x, y, Psi', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(vmax))
    f_over_H = [(1 + 0*y[j])/H([x[i], y[j]]) for i ∈ eachindex(x), j ∈ eachindex(y)]
    f_over_H = f_over_H.*mask
    ax.contour(x, y, f_over_H', colors="k", linewidths=0.5, alpha=0.5, linestyles="--", levels=get_levels(6))
    savefig(fname)
    println(fname)
    plt.close()
end

function profile_comparison(out_folders, labels, i_save)
    # plot name
    fname = @sprintf("%s/profiles%03d.png", out_folder, i_save)

    # setup points
    x = 0.5
    y = 0.0
    H0 = H([x, y])
    z = range(-H0, 0, length=2^8)
    points = [Point(x, y, zᵢ) for zᵢ ∈ z]

    # to overwrite later 
    t = 0.

    # plot
    fig, ax = plt.subplots(1, 3, figsize=(6, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"\partial_z b")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    for a ∈ ax 
        a.set_ylim(z[1], 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    for a ∈ ax[1:2]
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", linewidth=0.5, linestyle="-")
    end
    ax[3].set_xlim(0, 1.1)
    for i ∈ eachindex(out_folders)
        out_folder = out_folders[i]
        statefile = @sprintf("../sims/%s/data/state%03d.h5", out_folder, i_save)
        ux, uy, uz, p, b, t = load_state(statefile)
        ux = FEFunction(Ux, ux)
        uy = FEFunction(Uy, uy)
        uz = FEFunction(Uz, uz)
        b  = FEFunction(B, b)
        cache_ux = Gridap.CellData.return_cache(ux, points)
        cache_uy = Gridap.CellData.return_cache(uy, points)
        cache_uz = Gridap.CellData.return_cache(uz, points)
        cache_b  = Gridap.CellData.return_cache(b,  points)
        uxs = nan_eval(cache_ux, ux, points)
        uys = nan_eval(cache_uy, uy, points)
        uzs = nan_eval(cache_uz, uz, points)
        bs  = nan_eval(cache_b,  b,  points)
        dz = z[2] - z[1]
        bzs = (bs[3:end] - bs[1:end-2])/(2dz)
        bzs = [(-3/2*bs[1] + 2*bs[2] - 1/2*bs[3])/dz; bzs; (1/2*bs[end-2] - 2*bs[end-1] + 3/2*bs[end])/dz]
        bzs .+= 1
        uxs[1] = 0
        uys[1] = 0
        ax[1].plot(uxs[isnan.(uxs) .== 0], z[isnan.(uxs) .== 0], label=labels[i])
        ax[2].plot(uys[isnan.(uys) .== 0], z[isnan.(uys) .== 0])
        ax[3].plot(bzs[isnan.(bzs) .== 0], z[isnan.(bzs) .== 0])
    end
    ax[1].legend()
    if t === nothing
        ax[2].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f", x, y)))
    else
        log10t = floor(log10(t))
        ax[2].set_title(latexstring(@sprintf("x = %1.1f, \\quad y = %1.1f, \\quad t = %1.1f \\times 10^{%d} \\varepsilon^2/\\mu\\varrho", x, y, t/10^log10t, log10t)))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function profile_2D_vs_3D(fname2D)
    # read 2D data
    u2D, v2D, w2D, bz2D, z = load(fname2D, "u", "v", "w", "bz", "z")

    # shift z 
    z .+= 0.01

    # setup points
    x = 0.5
    y = 0.0
    points = [Point(x, y, zᵢ) for zᵢ ∈ z]

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
    ax[4].set_xlim(0, 1.1)
    cache_ux = Gridap.CellData.return_cache(ux, points)
    cache_uy = Gridap.CellData.return_cache(uy, points)
    cache_uz = Gridap.CellData.return_cache(uz, points)
    bz = 1 + ∂z(b)
    cache_bz  = Gridap.CellData.return_cache(bz, points)
    uxs = nan_eval(cache_ux, ux, points)
    uys = nan_eval(cache_uy, uy, points)
    uzs = nan_eval(cache_uz, uz, points)
    bzs = nan_eval(cache_bz, bz, points)
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0
    mask = isnan.(uxs) .== 0
    ax[1].plot(uxs[mask], z[mask])
    ax[1].plot(u2D, z, "k--", lw=0.5, label="2D")
    ax[2].plot(uys[mask], z[mask])
    ax[2].plot(v2D, z, "k--", lw=0.5)
    ax[3].plot(uzs[mask], z[mask])
    ax[3].plot(w2D, z, "k--", lw=0.5)
    ax[4].plot(bzs[mask], z[mask])
    ax[4].plot(bz2D, z, "k--", lw=0.5)
    ax[1].legend()
    ax[1].set_title(latexstring(@sprintf("x = %1.1f, \\quad y = %1.1f, \\quad t = %s", x, y, sci_notation(t))))
    savefig("$out_folder/images/profiles2Dvs3D.png")
    println("$out_folder/images/profiles2Dvs3D.png")
    plt.close()
end

function momentum_and_buoyancy_balance(out_folder, i_save)
    # parameters
    ε² = 1e-4
    f₀ = 1
    β = 0
    γ = 1/4
    μϱ = 1
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

    # profile
    x = 0.5
    y = 0.0
    H0 = H([x, y])
    z = range(-H0, 0, length=2^8)
    points = [Point(x, y, zᵢ) for zᵢ ∈ z]

    # load state file
    statefile = @sprintf("../sims/%s/data/state%03d.h5", out_folder, i_save)
    u, v, w, p, b, t = load_state(statefile)

    # define functions
    u = FEFunction(Ux, u)
    v = FEFunction(Uy, v)
    w = FEFunction(Uz, w)
    p  = FEFunction(P, p)
    b  = FEFunction(B, b)
    px = ∂x(p)
    py = ∂y(p)
    pz = ∂z(p)
    ux = ∂x(u)
    vy = ∂y(v)
    wz = ∂z(w)
    bx = ∂x(b)
    by = ∂y(b)
    bz = ∂z(b)
    # ∇²u = ε²*(γ*∂x(ν*∂x(u)) + γ*∂y(ν*∂y(u)) + ∂z(ν*∂z(u)))
    # uz_fef = FEFunction(Ux, ∂z(u))
    # ∇²u = ε²*∂z(uz_fef)

    # caches
    cache_u = Gridap.CellData.return_cache(u, points)
    cache_v = Gridap.CellData.return_cache(v, points)
    cache_w = Gridap.CellData.return_cache(w, points)
    cache_px = Gridap.CellData.return_cache(px,  points)
    cache_py = Gridap.CellData.return_cache(py,  points)
    cache_pz = Gridap.CellData.return_cache(pz,  points)
    cache_ux = Gridap.CellData.return_cache(ux,  points)
    cache_vy = Gridap.CellData.return_cache(vy,  points)
    cache_wz = Gridap.CellData.return_cache(wz,  points)
    cache_b  = Gridap.CellData.return_cache(b,  points)
    cache_bx = Gridap.CellData.return_cache(bx, points)
    cache_by = Gridap.CellData.return_cache(by, points)
    cache_bz = Gridap.CellData.return_cache(bz, points)
    # cache_∇²u = Gridap.CellData.return_cache(∇²u, points)

    # evals
    us = nan_eval(cache_u, u, points)
    vs = nan_eval(cache_v, v, points)
    ws = nan_eval(cache_w, w, points)
    pxs = nan_eval(cache_px, px, points)
    pys = nan_eval(cache_py, py, points)
    pzs = nan_eval(cache_pz, pz, points)
    uxs = nan_eval(cache_ux, ux, points)
    vys = nan_eval(cache_vy, vy, points)
    wzs = nan_eval(cache_wz, wz, points)
    bs  = nan_eval(cache_b,  b,  points)
    bxs = nan_eval(cache_bx, bx, points)
    bys = nan_eval(cache_by, by, points)
    bzs = nan_eval(cache_bz, bz, points)
    # ∇²us = nan_eval(cache_∇²u, ∇²u, points)

    # plot name
    fname = @sprintf("balances%03d.png", i_save)

    # plot
    fig, ax = plt.subplots(1, 5, figsize=(10, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"$x$-momentum")
    ax[2].set_xlabel(L"$y$-momentum")
    ax[3].set_xlabel(L"$z$-momentum")
    ax[4].set_xlabel("Continuity")
    ax[5].set_xlabel("Buoyancy")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])
    for a ∈ ax 
        a.set_ylim(z[1], 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", linewidth=0.5, linestyle="-")
    end
    # x momentum
    ax[1].plot(-(f₀ + β*y)*vs, z, label=L"-fv")
    ax[1].plot(-pxs, z, label=L"-\partial_x p")
    # ax[1].plot(∇²us, z, label=L"\varepsilon^2 \nabla \cdot (K_\nu \nabla u)")
    ax[1].legend()
    # y momentum
    ax[2].plot((f₀ + β*y)*us, z, label=L"fu")
    ax[2].plot(-pys, z, label=L"-\partial_y p")
    ax[2].legend()
    # z momentum
    ax[3].plot(bs, z, label=L"b")
    ax[3].plot(pzs, z, label=L"\partial_z p")
    ax[3].legend()
    # continuity
    ax[4].plot(uxs, z, label=L"\partial_x u")
    ax[4].plot(vys, z, label=L"\partial_y v")
    ax[4].plot(wzs, z, label=L"\partial_z w")
    ax[4].legend()
    # buoyancy
    ax[5].plot(us.*bxs, z, label=L"u \partial_x b")
    ax[5].plot(vs.*bys, z, label=L"v \partial_y b")
    ax[5].plot(ws.*(1 .+ bzs), z, label=L"w (1 + \partial_z b)")
    ax[5].legend()
    if t === nothing
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f", x, y)))
    else
        ax[1].set_title(latexstring(@sprintf("x = %1.1f, \\quad y = %d, \\quad t = %1.2f", x, y, t)))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

# for i_save ∈ 0:50
#     extract_profile_data(0.5, 0.0; i_save)
# end
# plot_profiles(["../sims/sim044/data/profiles010.jld2", 
#                "../sims/sim044/data/profiles020.jld2",
#                "../sims/sim044/data/profiles030.jld2",
#                "../sims/sim044/data/profiles040.jld2",
#                "../sims/sim035/data/profiles050.jld2"]; 
#                labels = ["010", "020", "030", "040", "050"],
#                fname="images/profiles_sim044.png")
# for i_save ∈ 0:50
for i_save ∈ [3]
    plot_profiles([@sprintf("../sims/sim044/data/profiles%03d.jld2", i_save), 
                   @sprintf("../sims/sim035/data/profiles%03d.jld2", i_save)]; 
                labels=["2D", "3D"], fname=@sprintf("images/profiles%03d.png", i_save))
end
# x, y, U, V, mask = compute_barotropic_velocities(ux, uy)
# Psi = compute_barotropic_streamfunction(U, y)
# jldsave("../out/data/psi037.jld2"; x, y, U, V, Psi, mask, i_save, t)
# plot_barotropic(x, y, U.*mask, V.*mask, Psi.*mask; i=i_save, t=t)
# plot_animation()
# profile_comparison(["sim038", "sim033", "sim030"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 3)
# profile_comparison(["sim035", "sim036", "sim037"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 3)
# profile_2D_vs_3D("../../PGModels1Dand2D/output/profilesPB1e-4.jld2")
# momentum_and_buoyancy_balance("sim033", 4)
