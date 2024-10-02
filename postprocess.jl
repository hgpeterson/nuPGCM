# using NonhydroPG
# using Gridap, GridapGmsh
# using Printf
# using PyPlot

# pygui(false)
# plt.style.use("plots.mplstyle")
# plt.close("all")

# # simulation folder
# out_folder = "sim033"

# # model
# hres = 0.01
# model = GmshDiscreteModel(@sprintf("meshes/bowl3D_%0.2f.msh", hres))

# # # full grid
# # m = Mesh(model)

# # # surface grid
# # m_sfc = Mesh(model, "sfc")

# # FE spaces
# X, Y, B, D = setup_FESpaces(model)
# Ux, Uy, Uz, P = unpack_spaces(X)

# # # triangulation
# # Ω = Triangulation(model)

# # depth
# H(x) = 1 - x[1]^2 - x[2]^2

# # load state file
# i_save = 10
# statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
# ux, uy, uz, p, b, t = load_state(statefile)
# ux = FEFunction(Ux, ux)
# uy = FEFunction(Uy, uy)
# uz = FEFunction(Uz, uz)
# p  = FEFunction(P, p)
# b  = FEFunction(B, b)

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
    for i_save ∈ 16:18
        statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
        ux, uy, uz, p, b, t = load_state(statefile)
        ux = FEFunction(Ux, ux)
        uy = FEFunction(Uy, uy)
        # uz = FEFunction(Uz, uz)
        b  = FEFunction(B, b)
        # plot_slice(ux, uz, b; y=0.0, t=t, cb_label=L"Speed $\sqrt{u^2 + w^2}$", cb_max=6e-2, fname=@sprintf("%s/images/uw_yslice_%03d.png", out_folder, i_save))
        # plot_slice(uy, b; y=0.0, t=t, cb_label=L"Meridional flow $v$", cb_max=1e-1, fname=@sprintf("%s/images/v_yslice_%03d.png", out_folder, i_save))
        # plot_slice(uz, b; z=-0.5, t=t, cb_label=L"Vertical flow $w$", cb_max=4e-3, fname=@sprintf("%s/images/w_zslice_%03d.png", out_folder, i_save))
        plot_slice(ux, uy, b; z=0.0, t=t, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", fname=@sprintf("%s/images/u_sfc_%03d.png", out_folder, i_save))
    end
end

function compare(out_folders, labels, i_save)
    # plot name
    fname = @sprintf("profiles%03d.png", i_save)

    # setup points
    x = 0.5
    y = 0.0
    H0 = H([x, y])
    z = range(-H0, 0, length=2^8)
    points = [Point(x, y, zᵢ) for zᵢ ∈ z]

    # to overwrite later 
    t = 0.

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
    for i ∈ eachindex(out_folders)
        out_folder = out_folders[i]
        statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
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
        uzs[1] = 0
        ax[1].plot(uxs, z, label=labels[i])
        ax[2].plot(uys, z)
        ax[3].plot(uzs, z)
        ax[4].plot(bzs, z)
    end
    ax[1].legend()
    if t === nothing
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f", x, y)))
    else
        ax[1].set_title(latexstring(@sprintf("x = %1.2f, \\quad y = %1.2f, \\quad t = %1.2f", x, y, t)))
    end
    savefig(fname)
    println(fname)
    plt.close()
end

# plot_animation()
compare(["sim034", "sim033", "sim030"], [L"\beta = 0.0", L"\beta = 0.5", L"\beta = 1.0"], 4)