using nuPGCM
using CUDA
using Gridap
using JLD2
using Printf
using Gridap

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing
# ENABLE_TIMING[] = true

PROJ_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM"
SIMS_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM/scratch/channel2D"

set_out_dir!(joinpath(SIMS_PATH, "sim003"))

# for making mesh
include(joinpath(PROJ_PATH, "meshes/mesh_channel2D.jl"))  

# architecture
arch = GPU()

# params
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
F₀ = N₀^4*H₀^4/f₀/L^2  # m² s⁻³

ε = sqrt(ν₀/f₀/H₀^2)
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2

t₀ = 1/f₀/ϱ  # s
@info "scales" b₀ ν₀ τ₀ t₀ F₀

μϱ = μ*ϱ
α = 1/8
N² = 0
f(x) = x[2]
H(x) = α
params = Parameters(; ε, α, μϱ, N², f, H)
display(params)

# forcings
κ_I = 1
κ_B = 1e2
d = 500/4000*α
ν(x) = 1
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
τˣ(x) = -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
# b_surface(x) = -b₀*x[2]^2
# b_surface_bc = SurfaceDirichletBC(b_surface)
b_flux_surface(x) = -1e-8/F₀*sin(2π*(x[2] + 1)/0.5)
b_surface_bc = SurfaceFluxBC(b_flux_surface)
# h_b = α/8
# b_basin(x) = -b₀ + (b_surface(x) + b₀)*exp(x[3]/h_b)
# b_basin(x) = -b₀ + (b_surface(x) + b₀)*(1 + x[3]/α)
b_basin(x) = b₀*x[3]/α
conv_param = ConvectionParameterization(κᶜ=0.2/κ₀, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=sqrt(1e-3))
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)
@info @sprintf("Diffusion timescale: %.2e", (κ_B * ε^2 / μϱ)^-1)

# mesh
h = 2.5e-3
mesh_name = @sprintf("channel2D_h%.2e_a%.2e", h, α)
mesh_file = joinpath(PROJ_PATH, "meshes/$mesh_name.msh")
if !isfile(mesh_file)
    generate_channel_mesh_2D(h, α)
end
mesh = Mesh(mesh_file)

# FE data
u_diri_tags  = ["bottom",           "coastline",        "basin bottom",     "basin top",         "surface",            "basin"]
u_diri_vals  = [(0, 0, 0),          (0, 0, 0),          (0, 0, 0),          (0, 0, 0),           (0, 0, 0),            (0, 0, 0)]
u_diri_masks = [(true, true, true), (true, true, true), (true, true, true), (false, true, true), (false, false, true), (false, true, false)]
if b_surface_bc isa SurfaceDirichletBC
    b_diri_tags = ["coastline", "surface", "basin top", "basin", "basin bottom"]
    b_diri_vals = [b_surface, b_surface, b_basin, b_basin, b_basin]
elseif b_surface_bc isa SurfaceFluxBC
    b_diri_tags = ["basin", "basin bottom"]
    b_diri_vals = [b_basin, b_basin]
end
spaces = Spaces(mesh; u_diri_tags, u_diri_vals, u_diri_masks, b_diri_tags, b_diri_vals, b_order=1) 
fe_data = FEData(mesh, spaces)
display(fe_data.dofs)

# setup inversion toolkit
inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; itmax=1000)

# set timestepper
Δt = 1*86400/t₀
t_stop = μϱ/ε^2/κ_I
timestepper = BDF1(; t_start=0, t_stop=t_stop, Δt=Δt, adaptive=true, CFL_factor=0.8)

# build evolution system
evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings, timestepper) 

# set up model
model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit, timestepper)

# set initial buoyancy
# set_b!(model, x -> b_surface(x)*exp(x[3]/(α/4)))
set_b!(model, x -> b₀*x[3]/α)
invert!(model)
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
@info @sprintf("Diffusion timescales: %.2e (κ_B), %.2e (κ_I)", μϱ/ε^2/κ_B, μϱ/ε^2/κ_I)
n_save = 100
run!(model; n_save)
