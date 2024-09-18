# using NonhydroPG
# using Gridap, GridapGmsh
# using Printf
# using PyPlot

# pygui(false)
# plt.style.use("plots.mplstyle")
# plt.close("all")

# # simulation folder
# out_folder = "sim024"

# # model
# hres = 0.05
# model = GmshDiscreteModel(@sprintf("meshes/bowl3D_%0.2f.msh", hres))

# # full grid
# m = Mesh(model)

# # surface grid
# m_sfc = Mesh(model, "sfc")

# # FE spaces
# X, Y, B, D = setup_FESpaces(model)
# Ux, Uy, Uz, P = unpack_spaces(X)

# # triangulation
# Ω = Triangulation(model)

# # depth
# H(x) = 1 - x[1]^2 - x[2]^2

# # load state file
# i_save = 25
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
plot_slice(ux, uy, b; z=0.0, t=t, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", fname=@sprintf("%s/images/uv_zslice_%03d.png", out_folder, i_save))
# plot_slice(ux, uz, b; y=0.0, t=t, cb_label=L"Speed $\sqrt{u^2 + w^2}$", fname=@sprintf("%s/images/uw_yslice_%03d.png", out_folder, i_save))
println()

function plot_animation()
    for i_save ∈ 1:24
        statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
        ux, uy, uz, p, b, t = load_state(statefile)
        # ux = FEFunction(Ux, ux)
        # uy = FEFunction(Uy, uy)
        uz = FEFunction(Uz, uz)
        b  = FEFunction(B, b)
        # plot_slice(ux, uz, b; y=0.0, t=t, cb_label=L"Speed $\sqrt{u^2 + w^2}$", cb_max=5e-3, fname=@sprintf("%s/images/uw_yslice_%03d.png", out_folder, i_save))
        # plot_slice(uy, b; y=0.0, t=t, cb_label=L"Meridional flow $v$", cb_max=1.5e-2, fname=@sprintf("%s/images/v_yslice_%03d.png", out_folder, i_save))
        plot_slice(uz, b; z=-0.5, t=t, cb_label=L"Vertical flow $w$", cb_max=4e-3, fname=@sprintf("%s/images/w_zslice_%03d.png", out_folder, i_save))
    end
end

# plot_animation()