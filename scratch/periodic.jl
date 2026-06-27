using nuPGCM
using Printf

ENV["JULIA_DEBUG"] = nuPGCM
ENABLE_TIMING[] = true

PROJ_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM-ferrite"
# SIMS_PATH = "/resnick/scratch/hppeters"
SIMS_PATH = @__DIR__

set_out_dir!(joinpath(SIMS_PATH, "periodic"))

# geom = :tub
geom = :box

# for making mesh
include(joinpath(PROJ_PATH, "meshes/periodic_rectangle.jl"))

# architecture
arch = CPU()

# params
ε = 1e-1
μϱ = 1
α = 1/4
N² = 0.0
f(x) = x[2]
H(x) = α
params = Parameters(; ε, α, μϱ, N², f, H)
display(params)

# resolution
h = 4e-2

# forcings
ν(x) = 1.0
κₕ(x) = 1 + (100 - 1)*exp(-(x[3] + H(x))/(α/4))
κᵥ(x) = 1 + (100 - 1)*exp(-(x[3] + H(x))/(α/4))
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0.0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
b_surface_bc = SurfaceDirichletBC(b_surface)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
display(forcings)
@info @sprintf("Diffusion timescale: %.2e", (κ_B * ε^2 / μϱ)^-1)

# mesh
mesh_name = @sprintf("periodic_rectangle_h%.2e_a%.2e", h, α)
mesh_file = joinpath(PROJ_PATH, "meshes/$mesh_name.msh")
if !isfile(mesh_file)
    mesh_periodic_rectangle(h, α)
end
mesh = Mesh(mesh_file)

# FE data
fe_data = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true, true, true), (false, false, true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [b_surface],
    b_order = 1)
display(fe_data)

# setup inversion toolkit
inv_tk = InversionToolkit(arch, fe_data, params, forcings)

# set timestepper
Δt = 1*86400/t₀
t_stop = μϱ/ε^2/κ_I
ts = BDF1(; t_start=0.0, t_stop, Δt, adaptive=true, CFL_factor=0.8)

# build evolution toolkit
evo_tk = EvolutionToolkit(arch, fe_data, params, forcings, ts)

# set up model
model = Model(arch, params, forcings, fe_data, inv_tk, evo_tk, ts)

# set initial buoyancy
# set_b!(model, x -> -b₀ + (b_surface(x) + b₀)*exp(x[3]/(α/4)))
set_b!(model, x -> x[3]/α)

invert!(model)
save_vtk(model; ofile="$out_dir/state_0000000000000000.vtu")

# # solve
# n_save = 100
# run!(model; n_save)