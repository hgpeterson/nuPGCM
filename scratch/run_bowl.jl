# Bowl with f = y (equator through the middle), bottom-enhanced mixing, no wind,
# fixed viscosity. GPU.
#
# NUPGCM_NSTEPS=<n> stops after n timesteps (setup check before a long job).
using nuPGCM
using CUDA, CUDSS
using Printf

ENV["JULIA_DEBUG"] = nuPGCM
ENABLE_TIMING[] = true

PROJ_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM-ferrite"
SIMS_PATH = "/home/hppeters/scratch"
set_out_dir!(joinpath(SIMS_PATH, "bowl_009"))

include(joinpath(PROJ_PATH, "meshes/mesh_bowl3D.jl"))

arch = GPU()

# params
ε = 2e-2
μϱ = 1.0
α = 1/2
N² = 0.0   # the stratification is carried by the initial b instead
f(x) = x[2]
H(x) = α*(1 - x[1]^2 - x[2]^2)
params = Parameters(; ε, α, μϱ, N², f, H)
display(params)

# resolution
h = 4e-2

# forcings
κ_I = 1e-2
κ_B = 1
d = 0.1
ν(x) = 1.0
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
b_surface(x) = 0.0
forcings = Forcings(ν, κₕ, κᵥ, x->0.0, x->0.0, SurfaceDirichletBC(b_surface))
display(forcings)

# mesh
mesh_file = joinpath(PROJ_PATH, @sprintf("meshes/bowl3D_%e_%e.msh", h, α))
isfile(mesh_file) || generate_bowl_mesh_3D(h, α)
mesh = Mesh(mesh_file)

# FE data
fe_data = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true, true, true), (false, false, true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [b_surface],
    b_order = 1)
display(fe_data)

inv_tk = InversionToolkit(arch, fe_data, params, forcings; itmax=400)

# Fixed Δt: the adaptive CFL rule is unstable for f = y (see simlog.txt).
# 2.4e-2 and 2.5e-3 agree to ~1%.
Δt = 2.4e-2
t_stop = μϱ/ε^2/κ_I
if haskey(ENV, "NUPGCM_NSTEPS")
    t_stop = Δt*parse(Int, ENV["NUPGCM_NSTEPS"])
    @warn "NUPGCM_NSTEPS set: short run, t_stop = $t_stop"
end
ts = BDF1(; t_start=0.0, t_stop, Δt)

evo_tk = EvolutionToolkit(arch, fe_data, params, forcings, ts)
model = Model(arch, params, forcings, fe_data, inv_tk, evo_tk, ts)

set_b!(model, x -> x[3]/α)
invert!(model)

@info @sprintf("Diffusion timescales: %.2e (κ_B), %.2e (κ_I)", μϱ/ε^2/κ_B, μϱ/ε^2/κ_I)
run!(model; n_save=100)
