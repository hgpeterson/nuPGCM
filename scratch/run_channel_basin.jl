# Channel + basin (flat-bottom box), f = y, wind + surface buoyancy forcing,
# convection and eddy parameterizations on. GPU.
#
# NUPGCM_NSTEPS=<n> stops after n timesteps (setup check before a long job).
using nuPGCM
using CUDA, CUDSS
using Printf

ENV["JULIA_DEBUG"] = nuPGCM
ENABLE_TIMING[] = true

PROJ_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM-ferrite"
SIMS_PATH = "/home/hppeters/scratch"
set_out_dir!(joinpath(SIMS_PATH, "channel_basin_013"))

include(joinpath(PROJ_PATH, "meshes/channel_basin_flat.jl"))

arch = GPU()

# dimensional scales
Ω = 2π/86400  # s⁻¹
a = 6.371e6  # m
β = 2Ω/a  # m⁻¹ s⁻¹
L = 2π*a*60/360  # m
f₀ = β*L  # s⁻¹
H₀ = 4e3  # m
κ₀ = 1e-5  # m² s⁻¹
Kₑ = 1000  # m² s⁻¹
N₀ = 1e-3  # s⁻¹
ρ₀ = 1035  # kg m⁻³
α_T = 2e-4  # °C⁻¹
g = 9.81  # m s⁻²
ν₀ = Kₑ*f₀^2/N₀^2  # m² s⁻¹
τ₀ = ρ₀*N₀^2*H₀^3/L  # N m⁻²
b₀ = g*α_T*30/(N₀^2*H₀)
t₀ = 1/f₀/(N₀*H₀/f₀/L)^2  # s
@info "scales" b₀ ν₀ τ₀ t₀

# params
ε = sqrt(ν₀/f₀/H₀^2)
μϱ = ν₀/κ₀*(N₀*H₀/f₀/L)^2
α = 1/4
N² = 0.0
f(x) = x[2]
H(x) = α
params = Parameters(; ε, α, μϱ, N², f, H)
display(params)

# resolution
h = 4e-2

# forcings
κ_I = 5.706e+00
κ_B = 2.535e+01
d = 3.526e-01*α
ν(x) = 1.0
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0.0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
conv_param = ConvectionParameterization(κᶜ=0.2/κ₀, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=sqrt(1e-3), ν_min=0.1)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, SurfaceDirichletBC(b_surface); conv_param, eddy_param)
display(forcings)

# mesh
mesh_file = joinpath(PROJ_PATH, @sprintf("meshes/channel_basin_flat_h%.2e_a%.2e.msh", h, α))
isfile(mesh_file) || mesh_channel_basin_flat(h, α)
mesh = Mesh(mesh_file)

# FE data
fe_data = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true, true, true), (false, false, true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [b_surface],
    b_order = 1)
display(fe_data)

# inversion: converged solves take 107-274 iterations here, so 400 caps the cost
# of a miss (run! rebuilds a stale preconditioner on the next step)
inv_tk = InversionToolkit(arch, fe_data, params, forcings; itmax=400)

# Adaptive Δt capped at 4e-4: the CFL limit alone is unstable for f = y (see
# simlog.txt). Fixed BDF1 blew up at 8e-4 by step ~50; 4e-4, 2e-4 and 1e-4 agree
# to 0.4% at matched t.
Δt = 4e-4
t_stop = μϱ/ε^2/κ_I
if haskey(ENV, "NUPGCM_NSTEPS")
    t_stop = Δt*parse(Int, ENV["NUPGCM_NSTEPS"])
    @warn "NUPGCM_NSTEPS set: short run, t_stop = $t_stop"
end
ts = BDF2(; t_start=0.0, t_stop, Δt, adaptive=true)

evo_tk = EvolutionToolkit(arch, fe_data, params, forcings, ts)
model = Model(arch, params, forcings, fe_data, inv_tk, evo_tk, ts)

# initial condition (to restart instead: set_state_from_file!(model, file) and
# pass i_start to run!)
set_b!(model, x -> -b₀ + (b_surface(x) + b₀)*exp(x[3]/(α/4)))
invert!(model)

@info @sprintf("Diffusion timescales: %.2e (κ_B), %.2e (κ_I)", μϱ/ε^2/κ_B, μϱ/ε^2/κ_I)
run!(model; n_save=100)
