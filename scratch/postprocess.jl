using nuPGCM
using Gridap, GridapGmsh
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

# simulation folder
out_folder = "../sims/sim035"

# # model
# hres = 0.01
# model = GmshDiscreteModel(@sprintf("../meshes/bowl3D_%0.2f.msh", hres))

# # FE spaces
# X, Y, B, D = setup_FESpaces(model)
# Ux, Uy, Uz, P = unpack_spaces(X)

# # triangulation
# Ω = Triangulation(model)

# depth
H(x) = 1 - x[1]^2 - x[2]^2

# load state file
i_save = 3
statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
ux, uy, uz, p, b, t = load_state(statefile)
ux = FEFunction(Ux, ux)
uy = FEFunction(Uy, uy)
uz = FEFunction(Uz, uz)
p  = FEFunction(P, p)
b  = FEFunction(B, b)

# # save vtu
# save_state_vtu(ux, uy, uz, p, b, Ω; fname=@sprintf("%s/data/state%03d.vtu", out_folder, i_save))

# plot_slice(ux, b; x=0,    t=t, cb_label=L"Zonal flow $u$", fname=@sprintf("%s/images/u_xslice_%03d.png", out_folder, i_save))
plot_slice(ux, b; y=0,    t=t, cb_label=L"Zonal flow $u$", fname=@sprintf("%s/images/u_yslice_%03d.png", out_folder, i_save))
# plot_slice(ux, b; z=-0.5, t=t, cb_label=L"Zonal flow $u$", fname=@sprintf("%s/images/u_zslice_%03d.png", out_folder, i_save))
# plot_slice(uy, b; x=0,    t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_xslice_%03d.png", out_folder, i_save))
plot_slice(uy, b; y=0,    t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_yslice_%03d.png", out_folder, i_save))
# plot_slice(uy, b; z=-0.5, t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v_zslice_%03d.png", out_folder, i_save))
# plot_slice(uz, b; x=0,    t=t, cb_label=L"Vertical flow $w$", fname=@sprintf("%s/images/w_xslice_%03d.png", out_folder, i_save))
# plot_slice(uz, b; y=0,    t=t, cb_label=L"Vertical flow $w$", fname=@sprintf("%s/images/w_yslice_%03d.png", out_folder, i_save))
# plot_slice(uz, b; z=-0.5, t=t, cb_label=L"Vertical flow $w$", fname=@sprintf("%s/images/w_zslice_%03d.png", out_folder, i_save))
plot_slice(ux, uy, b; z=0.0, t=t, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", fname=@sprintf("%s/images/uv_zslice_%03d.png", out_folder, i_save))
# plot_slice(ux, uz, b; y=0.0, t=t, cb_label=L"Speed $\sqrt{u^2 + w^2}$", fname=@sprintf("%s/images/uw_yslice_%03d.png", out_folder, i_save))
# println()

function plot_animation()
    for i_save ∈ [50]
        statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
        ux, uy, uz, p, b, t = load_state(statefile)
        ux = FEFunction(Ux, ux)
        uy = FEFunction(Uy, uy)
        # uz = FEFunction(Uz, uz)
        b  = FEFunction(B, b)
        plot_slice(ux, b;     y=0, t=t, cb_label=L"Zonal flow $u$",                      cb_max=6e-3,   fname=@sprintf("%s/images/u_yslice_%03d.png", out_folder, i_save))
        plot_slice(uy, b;     y=0, t=t, cb_label=L"Meridional flow $v$",                 cb_max=1.5e-1, fname=@sprintf("%s/images/v_yslice_%03d.png", out_folder, i_save))
        plot_slice(ux, uy, b; z=0, t=t, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", cb_max=2e-1,   fname=@sprintf("%s/images/u_sfc_%03d.png", out_folder, i_save))
    end
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

∂x(u) = VectorValue(1, 0, 0)⋅∇(u)
∂y(u) = VectorValue(0, 1, 0)⋅∇(u)
∂z(u) = VectorValue(0, 0, 1)⋅∇(u)

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

# plot_animation()
# profile_comparison(["sim038", "sim033", "sim030"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 3)
# profile_comparison(["sim035", "sim036", "sim037"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 3)
# momentum_and_buoyancy_balance("sim033", 4)
