using nuPGCM
using Printf

# Multi-step advection test: flat isopycnal initial condition, invert+evolve
# each step. Prints max|u| and max|b| every n_info steps and saves VTU snapshots.

set_out_dir!(joinpath(@__DIR__, "bowl_advection_out"))

α  = 0.5
ε  = 2e-1
μϱ = 1e1
N² = 1/α
f(x)  = 1.0 + 0.5*x[2]
H(x)  = α*(1 - x[1]^2 - x[2]^2)
params = Parameters(; ε, α, μϱ, N², f, H)

κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
forcings = Forcings(1.0, κ, κ, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))

bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
mesh      = Mesh(bowl_file)
fe_data   = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true,true,true), (false,false,true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [x -> 0.0])

Δt      = 1e-1
n_steps = 50
t_stop  = n_steps * Δt

inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
ts     = BDF1(; t_start=0.0, t_stop, Δt)
evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
model  = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)

set_b!(model, x -> 0)

run!(model; n_save=10)