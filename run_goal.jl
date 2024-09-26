using NonhydroPG
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

# set output folder
out_folder = "sim030"
set_out_folder!(out_folder)

# choose architecture
# arch = CPU()
arch = GPU()

# tolerance and max iterations for iterative solvers
tol = 1e-8
@printf("tol = %.1e\n", tol)
itmax = 0
@printf("itmax = %d\n", itmax)

# geometry
hres = 0.01
mesh_file = @sprintf("meshes/bowl%s_%0.2f.msh", dim, hres)
# mesh_file = "bowl2D_exp.msh"
geometry = Geometry(mesh_file)

# forcing
H(x) = 1 - x[1]^2 - x[2]^2
ν(x) = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)
forcing = Forcing(ν, κ)

# params
ε² = 1e-4
γ = 1/4
f₀ = 1
β = 1
f(x) = f₀ + β*x[2]
μϱ = 1e0
Δt = 0.05
T = 5e-2*μϱ/ε²
parameters = Parameters(ε², γ, f, μϱ, Δt, T)

# folder where LHS matrices are stored
matix_folder = "matrices/"

# model
model = Model(dim, arch, geometry, forcing, parameters, matix_folder)

# # initial condition: b = 0, t = 0
# i_save = 0
# b(x) = 0
# t = 0.
# set_state!(model, b, t)
# save(model.state; fname=@sprintf("%s/data/state%03d.jld2", out_folder, i_save))

# initial condition: load from file
i_save = 20
set_state!(model, @sprintf("%s/data/state%03d.jld2", out_folder, i_save))

# plot initial condition and generate plots cache
plots_cache = sim_plots(model.state, out_folder, i_save)
i_save += 1

# run
solve!(model)