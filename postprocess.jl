using NonhydroPG
using Gridap, GridapGmsh
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

out_folder = "sim022"

# model
hres = 0.01
model = GmshDiscreteModel(@sprintf("meshes/bowl3D_%0.2f.msh", hres))

# full grid
m = Mesh(model)

# surface grid
m_sfc = Mesh(model, "sfc")

# FE spaces
X, Y, B, D = setup_FESpaces(model)
Ux, Uy, Uz, P = unpack_spaces(X)

# triangulation
Ω = Triangulation(model)

# depth
H(x) = 1 - x[1]^2 - x[2]^2

# load state file
i_save = 23
statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
ux, uy, uz, p, b, t = load_state(statefile)
ux = FEFunction(Ux, ux)
uy = FEFunction(Uy, uy)
uz = FEFunction(Uz, uz)
p  = FEFunction(P, p)
b  = FEFunction(B, b)

# save vtu
save_state_vtu(ux, uy, uz, p, b, Ω; fname=@sprintf("%s/data/state%03d.vtu", out_folder, i_save))

# plot slice
plot_yslice(uy, b, 0, H; t=t, cb_label=L"Meridional flow $v$", fname=@sprintf("%s/images/v%03d.png", out_folder, i_save))