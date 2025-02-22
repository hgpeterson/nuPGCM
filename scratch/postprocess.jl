using nuPGCM
using Gridap
using GridapGmsh
using Printf
using PyPlot
using JLD2

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

include("../plots/derivatives.jl")

# # simulation folder
# set_out_dir!("../sims/sim048")

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
# Hx(x) = -2x[1]
# Hy(x) = -2x[2]

# # stratification
# N² = 1

# # load state file
# i_save = 3
# statefile = @sprintf("%s/data/state%03d.h5", out_dir, i_save)
# ux, uy, uz, p, b, t = load_state(statefile)
# ux = FEFunction(Ux, ux)
# uy = FEFunction(Uy, uy)
# uz = FEFunction(Uz, uz)
# p  = FEFunction(P, p)
# b  = FEFunction(B, b)

################################################################################

function save_gridded_data(statefile; n=2^6, outfile="gridded.jld2")
    # load state file
    ux, uy, uz, p, b, t = load_state(statefile)
    ux = FEFunction(Ux, ux)
    uy = FEFunction(Uy, uy)
    uz = FEFunction(Uz, uz)
    p  = FEFunction(P, p)
    b  = FEFunction(B, b)

    # cartesian grid
    if mod(n, 2) == 0 
        @warn "n must be odd, incrementing to $n + 1"
        n += 1
    end
    x = range(-1, 1, length=n)
    y = range(-1, 1, length=n)
    z = range(-1, 0, length=n)
    points = [Point(x[i], y[j], z[k]) for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n][:]

    # evaluate fields
    @time "evaluate u" u = reshape(nan_eval(ux, points), n, n, n)
    @time "evaluate v" v = reshape(nan_eval(uy, points), n, n, n)
    @time "evaluate w" w = reshape(nan_eval(uz, points), n, n, n)
    @time "evaluate b" b = reshape(nan_eval(b,  points), n, n, n)

    # save data
    jldsave(outfile; x, y, z, u, v, w, b, t)
    @info "Data saved to '$outfile'"
end

function load_gridded_data(file)
    jldopen(file) do file
        x = file["x"]
        y = file["y"]
        z = file["z"]
        u = file["u"]
        v = file["v"]
        w = file["w"]
        b = file["b"]
        t = file["t"]
        return x, y, z, u, v, w, b, t
    end
end

function save_gridded_sigma_data(statefile; n=2^6, outfile="gridded_sigma.jld2")
    # load state file
    ux, uy, uz, p, b, t = load_state(statefile)
    ux = FEFunction(Ux, ux)
    uy = FEFunction(Uy, uy)
    uz = FEFunction(Uz, uz)
    p  = FEFunction(P, p)
    b  = FEFunction(B, b)

    # depth function
    H(x) = 1 - x[1]^2 - x[2]^2

    # horizontal grid
    if mod(n, 2) == 0 
        @warn "n must be odd, incrementing to $n + 1"
        n += 1
    end
    x = range(-1, 1, length=n)
    y = range(-1, 1, length=n)
    H = [H([x[i], y[j]]) for i ∈ 1:n, j ∈ 1:n]

    # vertical sigma grid
    σ = range(-1, 0, length=n)

    # points
    points = [Point(x[i], y[j], σ[k]*H[i, j]) for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n][:]

    # evaluate fields
    @time "evaluate u" u = reshape(nan_eval(ux, points), n, n, n)
    @time "evaluate v" v = reshape(nan_eval(uy, points), n, n, n)
    @time "evaluate w" w = reshape(nan_eval(uz, points), n, n, n)
    @time "evaluate b" b = reshape(nan_eval(b,  points), n, n, n)

    # save data
    jldsave(outfile; x, y, σ, u, v, w, b, H, t)
    @info "Data saved to '$outfile'"
end

function load_gridded_sigma_data(file)
    jldopen(file) do file
        x = file["x"]
        y = file["y"]
        σ = file["σ"]
        u = file["u"]
        v = file["v"]
        w = file["w"]
        b = file["b"]
        H = file["H"]
        t = file["t"]
        return x, y, σ, u, v, w, b, H, t
    end
end

# for contour lines
get_levels(vmax) = [-vmax, -3vmax/4, -vmax/2, -vmax/4, vmax/4, vmax/2, 3vmax/4, vmax]

function extract_profile_data(x, y; i_save=0)
    # load state file
    statefile = @sprintf("%s/data/state%03d.h5", out_dir, i_save)
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
    jldsave(@sprintf("%s/data/profiles%03d.jld2", out_dir, i_save); x, y, z, t, uxs, uys, uzs, ps, bs, bzs, ux_mask, uy_mask, uz_mask, p_mask, bz_mask)
    println(@sprintf("%s/data/profiles%03d.jld2", out_dir, i_save))
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

function compute_U_V(u, v, σ, H)
    # integrate
    U = [trapz(u[i, j, :], σ*H[i, j]) for i in axes(u, 1), j in axes(u, 2)]
    V = [trapz(v[i, j, :], σ*H[i, j]) for i in axes(u, 1), j in axes(u, 2)]

    # NaNs outside of domain
    for i in axes(u, 1), j in axes(u, 2)
        if all(isnan.(u[i, j, :]))
            U[i, j] = NaN
            V[i, j] = NaN
        end
    end

    # # debug: number of NaNs per column
    # nan_count = [sum(isnan.(u[i, j, :])) for i in axes(u, 1), j in axes(u, 2)]
    # # imshow(log.(nan_count/size(u, 1)))
    # # colorbar(label="log(NaN fraction)")
    # imshow(nan_count, vmax=10)
    # colorbar()
    # savefig("nan_count.png")
    # @info "Saved 'nan_count.png'"
    # plt.close()

    return U, V
end

function compute_Ψ(U, y)
    Ψ = zeros(size(U))
    for i in axes(U, 1), j in axes(U, 2)
        if isnan(U[i, j]) || isnan(U[i, j-1])
            Ψ[i, j] = NaN
        elseif isnan(Ψ[i, j-1])
            Ψ[i, j] = -0.5*(y[j] - y[j-1])*(U[i, j] + U[i, j-1])
        else
            Ψ[i, j] = Ψ[i, j-1] - 0.5*(y[j] - y[j-1])*(U[i, j] + U[i, j-1])
        end
        # Ψ[i, :] = -cumtrapz(U[i, :], y)
    end
    return Ψ
end

function plot_barotropic(x, y, F; fig=nothing, ax=nothing, label=L"F")
    if ax === nothing
        fig, ax = plt.subplots(1)
    end
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks(-1:1:1)
    ax.set_yticks(-1:1:1)
    vmax = nan_max(abs.(F))
    img = ax.pcolormesh(x, y, F', shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x, y, F', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(vmax))
    return fig, ax
end

function add_title(ax, t)
    ax.set_title(latexstring(@sprintf("t = %s", sci_notation(t))))
end

function save_plot(fname_base; i=0)
    if i == 0
        fname = @sprintf("%s/images/%s.png", out_dir, fname_base)
    else
        fname = @sprintf("%s/images/%s%03d.png", out_dir, fname_base, i)
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_UV_Ψ(x, y, U, V, Ψ; i=0, t=nothing)
    # plot UV
    fig, ax = plt.subplots(1, 2, figsize=(3.2*2, 2))
    plot_barotropic(x, y, U', fig=fig, ax=ax[1], label=L"U")
    plot_barotropic(x, y, V', fig=fig, ax=ax[2], label=L"V")
    if t !== nothing
        add_title(ax[1], t)
        add_title(ax[2], t)
    end
    subplots_adjust(wspace=0.4)
    save_plot("UV"; i)

    # plot Psi
    fig, ax = plot_barotropic(x, y, Ψ', label=L"\Psi")
    f_over_H = [(1 + 0*y[j])/H[i, j] for i ∈ eachindex(x), j ∈ eachindex(y)]
    ax.contour(x, y, f_over_H', colors="k", linewidths=0.5, alpha=0.5, linestyles="--", levels=get_levels(6))
    save_plot("psi"; i)
end

function compute_γ(b)
    # resolution
    nx = 2^5
    ny = nx
    nz = 2^5

    # 3D grid
    x = range(-1, 1, length=nx)
    y = range(-1, 1, length=ny)
    σ = chebyshev_nodes(nz)

    # points 
    points = [Point(x[i], y[j], H([x[i], y[j]])*σ[k]) for i ∈ 1:nx, j ∈ 1:ny, k ∈ 1:nz][:]

    # evaluate field
    @time "evals" bs = reshape(nan_eval(b, points), nx, ny, nz)

    # -∫ z*b dz
    γ = -[trapz(H([x[i], y[j]])*σ.*bs[i, j, :], H([x[i], y[j]])*σ) for i ∈ 1:nx, j ∈ 1:ny]

    # NaNs outside of domain
    for i ∈ 1:nx, j ∈ 1:ny
        if all(isnan.(bs[i, j, :]))
            γ[i, j] = NaN
        end
    end

    return x, y, γ
end

function compute_JEBAR(x, y, γ)
    nx = length(x)
    ny = length(y)
    γx = zeros(nx, ny)
    γy = zeros(nx, ny)
    for i ∈ 1:nx
        γy[i, :] = differentiate(γ[i, :], y)
    end
    for j ∈ 1:ny
        γx[:, j] = differentiate(γ[:, j], x)
    end
    # -J(1/H, γ) = (Hx*γy - Hy*γx)/H^2
    return [(Hx([x[i], y[j]])*γy[i, j] - Hy([x[i], y[j]])*γx[i, j])/H([x[i], y[j]])^2 for i ∈ 1:nx, j ∈ 1:ny]
end

function profile_comparison(out_dirs, labels, i_save)
    # plot name
    fname = @sprintf("%s/profiles%03d.png", out_dir, i_save)

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
    for i ∈ eachindex(out_dirs)
        out_dir = out_dirs[i]
        statefile = @sprintf("../sims/%s/data/state%03d.h5", out_dir, i_save)
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
    savefig("$out_dir/images/profiles2Dvs3D.png")
    println("$out_dir/images/profiles2Dvs3D.png")
    plt.close()
end

function momentum_and_buoyancy_balance(out_dir, i_save)
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
    statefile = @sprintf("../sims/%s/data/state%03d.h5", out_dir, i_save)
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


function compute_BVE_buoyancy_source(n)
    # load data
    i_save = 3
    file = jldopen(@sprintf("%s/data/gridded_sigma_n%04d_i%03d.jld2", out_dir, n, i_save))
    x = file["x"]
    y = file["y"]
    σ = file["σ"]
    # u = file["u"]
    # v = file["v"]
    # w = file["w"]
    b = file["b"]
    H = file["H"]
    t = file["t"]
    close(file)

    # differentiate H
    Hx = zeros(size(H))
    Hy = zeros(size(H))
    for i ∈ 1:n
        Hx[:, i] = differentiate(H[:, i], x)
        Hy[i, :] = differentiate(H[i, :], y)
    end

    # # debug: compare with analytical -2x and -2y
    # Hx_error = [Hx[i, j] - (-2*x[i]) for i ∈ 1:n, j ∈ 1:n]
    # Hy_error = [Hy[i, j] - (-2*y[j]) for i ∈ 1:n, j ∈ 1:n]
    # fig, ax = plot_barotropic(x, y, Hx_error; label=L"\partial_x H")
    # add_title(ax, t)
    # save_plot("Hx_error"; i=i_save)
    # fig, ax = plot_barotropic(x, y, Hy_error; label=L"\partial_y H")
    # add_title(ax, t)
    # save_plot("Hy_error"; i=i_save)

    # differentiate b
    bξ = zeros(n, n, n)
    bη = zeros(n, n, n)
    bσ = zeros(n, n, n)
    for i ∈ 1:n, j ∈ 1:n 
        bξ[:, i, j] = differentiate(b[:, i, j], x)
        bη[i, :, j] = differentiate(b[i, :, j], y)
        bσ[i, j, :] = differentiate(b[i, j, :], σ)
    end
    bx = [bξ[i, j, k] - σ[k]*Hx[i, j]/H[i, j]*bσ[i, j, k] for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n]
    by = [bη[i, j, k] - σ[k]*Hy[i, j]/H[i, j]*bσ[i, j, k] for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n]

    # compute drag coefficient
    ν_bot = 1
    β = 0
    f = [1 + β*y[j] for i ∈ 1:n, j ∈ 1:n]
    q = @. √(2ν_bot/f)
    r = ν_bot*q./H

    # compute terms 1 and 2
    term1 = [r[i, j]/f[i, j]*H[i, j] * trapz(σ.*(bx[i, j, :] - by[i, j, :]), σ) for i ∈ 1:n, j ∈ 1:n]
    term2 = [r[i, j]/f[i, j]*H[i, j] * trapz(σ.*(bx[i, j, :] + by[i, j, :]), σ) for i ∈ 1:n, j ∈ 1:n]

    # plot terms 1 and 2
    fig, ax = plot_barotropic(x, y, term1; label="term 1")
    add_title(ax, t)
    save_plot("term1_"; i=i_save)
    fig, ax = plot_barotropic(x, y, term2; label="term 2")
    add_title(ax, t)
    save_plot("term2_"; i=i_save)

    # compute convergence
    BVE_buoyancy_source = zeros(n, n)
    for i ∈ 1:n
        BVE_buoyancy_source[:, i] .-= differentiate(term1[:, i], x)
        BVE_buoyancy_source[i, :] .-= differentiate(term2[i, :], y)
    end

    # plot BVE buoyancy source
    fig, ax = plot_barotropic(x, y, BVE_buoyancy_source; label=L"\mathcal{B}")
    add_title(ax, t)
    save_plot("BVE_buoyancy_source"; i=i_save)

    # plot BVE buoyancy source just along y=0
    fig, ax = plt.subplots(1)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"\mathcal{B}")
    # ax.plot(x, BVE_buoyancy_source[:, n÷2])
    ax.plot(x, BVE_buoyancy_source[:, n÷2] .* (sqrt(2)*(1 .- x.^2).^2))
    println(y[n÷2])
    savefig(@sprintf("%s/images/BVE_buoyancy_source_y0_%03d.png", out_dir, i_save))
    println(@sprintf("%s/images/BVE_buoyancy_source_y0_%03d.png", out_dir, i_save))
    plt.close()

    return x, BVE_buoyancy_source
end

# sim_dir = "../sims/sim048"
# i_save = 3
# n = 2^8 + 1
# save_gridded_sigma_data(@sprintf("%s/data/state_beta0.0_%03d.h5", sim_dir, i_save); n=n, outfile=@sprintf("%s/data/gridded_sigma_beta0.0_n%04d_i%03d.jld2", sim_dir, n, i_save))
# save_gridded_sigma_data(@sprintf("%s/data/state_beta0.5_%03d.h5", sim_dir, i_save); n=n, outfile=@sprintf("%s/data/gridded_sigma_beta0.5_n%04d_i%03d.jld2", sim_dir, n, i_save))
# save_gridded_sigma_data(@sprintf("%s/data/state_beta1.0_%03d.h5", sim_dir, i_save); n=n, outfile=@sprintf("%s/data/gridded_sigma_beta1.0_n%04d_i%03d.jld2", sim_dir, n, i_save))

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
# for i_save ∈ [3]
#     plot_profiles([@sprintf("../sims/sim044/data/profiles%03d.jld2", i_save), 
#                    @sprintf("../sims/sim035/data/profiles%03d.jld2", i_save)]; 
#                 labels=["2D", "3D"], fname=@sprintf("images/profiles%03d.png", i_save))
# end

x, y, σ, u, v, w, b, H, t = load_gridded_sigma_data("../sims/sim048/data/gridded_sigma_beta0.0_n0257_i003.jld2")
U, V = compute_U_V(u, v, σ, H)
Ψ = compute_Ψ(U, y)
plot_UV_Ψ(x, y, U, V, Ψ; i=3, t=t)
# plot_barotropic(x, y, U; i=i_save, t=t)
# x, y, U, V, mask = compute_barotropic_velocities(ux, uy)
# Psi = compute_barotropic_streamfunction(U, y)
# jldsave("../out/data/psi037.jld2"; x, y, U, V, Psi, mask, i_save, t)
# plot_barotropic(x, y, U.*mask, V.*mask, Psi.*mask; i=i_save, t=t)

# x, y, γ = compute_γ(b)
# fig, ax = plot_barotropic(x, y, γ; label=L"\gamma")
# add_title(ax, t)
# save_plot("gamma"; i=i_save)

# JEBAR = compute_JEBAR(x, y, γ)
# fig, ax = plot_barotropic(x, y, JEBAR; label=L"-J(1/H, \gamma)")
# add_title(ax, t)
# save_plot("JEBAR"; i=i_save)

# plot_animation()
# profile_comparison(["sim038", "sim033", "sim030"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 3)
# profile_comparison(["sim035", "sim036", "sim037"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 3)
# profile_2D_vs_3D("../../PGModels1Dand2D/output/profilesPB1e-4.jld2")
# momentum_and_buoyancy_balance("sim033", 4)

# x_4, BVE_buoyancy_source_4 = compute_BVE_buoyancy_source(2^4)
# x_5, BVE_buoyancy_source_5 = compute_BVE_buoyancy_source(2^5)
# x_6, BVE_buoyancy_source_6 = compute_BVE_buoyancy_source(2^6)
# x_7, BVE_buoyancy_source_7 = compute_BVE_buoyancy_source(2^7)
# x_8, BVE_buoyancy_source_8 = compute_BVE_buoyancy_source(2^8)

# fig, ax = plt.subplots(1)
# ax.set_xlabel(L"x")
# ax.set_title(L"\mathcal{B}(x, 0) \sqrt{2} (1 - x^2)^2")
# # ax.set_xlim(0, 1)
# ax.set_ylim(-1, 1)
# ax.spines["bottom"].set_visible(false)
# ax.axhline(0, color="k", linewidth=0.5, linestyle="-")
# ax.plot(x_4, BVE_buoyancy_source_4[:, 2^3] .* (sqrt(2)*(1 .- x_4.^2).^2), label=L"n=2^4")
# ax.plot(x_5, BVE_buoyancy_source_5[:, 2^4] .* (sqrt(2)*(1 .- x_5.^2).^2), label=L"n=2^5")
# ax.plot(x_6, BVE_buoyancy_source_6[:, 2^5] .* (sqrt(2)*(1 .- x_6.^2).^2), label=L"n=2^6")
# ax.plot(x_7, BVE_buoyancy_source_7[:, 2^6] .* (sqrt(2)*(1 .- x_7.^2).^2), label=L"n=2^7")
# ax.plot(x_8, BVE_buoyancy_source_8[:, 2^7] .* (sqrt(2)*(1 .- x_8.^2).^2), label=L"n=2^8")
# ax.legend(ncol=2)
# i_save = 3
# savefig(@sprintf("%s/images/BVE_buoyancy_source_y0_%03d.png", out_dir, i_save))
# println(@sprintf("%s/images/BVE_buoyancy_source_y0_%03d.png", out_dir, i_save))
# plt.close()