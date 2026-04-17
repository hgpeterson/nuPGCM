using nuPGCM
using CUDA
using Gridap
using JLD2
using Printf
using Gridap

# for making mesh
# include(joinpath(@__DIR__, "../meshes/channel_basin_no_flat_round_end.jl"))  
include(joinpath(@__DIR__, "../meshes/channel_basin_flat.jl"))  

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

set_out_dir!("/resnick/scratch/hppeters/sim051e")

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

ε = sqrt(ν₀/f₀/H₀^2)
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2

t₀ = 1/f₀/ϱ  # s
@info "scales" b₀ ν₀ τ₀ t₀

μϱ = μ*ϱ
α = 1/4
N² = 0
f(x) = x[2]
H(x) = α
# function H((x, y, z))
#     L = 2
#     W = 1
#     L_channel = L/4
#     L_flat_channel = 5L_channel/8 # length of flat part of channel
#     H = α*W

#     # parabola that has a maximum of H at x_max and a 0 at x_zero
#     parabola(x, x_max, x_zero) = H*(1 - ((x - x_max)/(x_zero - x_max))^2)

#     function H_basin(x)
#         if 0 ≤ x ≤ W
#             return parabola(x, W/2, 0)
#         else
#             throw(ArgumentError("x out of bounds"))
#         end
#     end

#     if -L/2 ≤ y ≤ -L/2 + L_flat_channel
#         return H
#     elseif y ≤ -L/2 + L_channel
#         H_channel = parabola(y, -L/2 + L_flat_channel, -L/2 + L_channel)
#         return max(H_channel, H_basin(x))
#     elseif y ≤ L/2 - W/2
#         return H_basin(x)
#     elseif y ≤ L/2
#         r = √( (x - W/2)^2 + (y - (L/2 - W/2))^2 )
#         if r > W/2
#             if r - W/2 < 1e-1 # points on boundary might just need a fudge factor
#                 return 0
#             else
#                 throw(ArgumentError("(x, y) out of bounds"))
#             end
#         else
#             return parabola(r, 0, W/2)
#         end
#     else
#         throw(ArgumentError("y out of bounds"))
#     end
# end
params = Parameters(; ε, α, μϱ, N², f, H)
display(params)

# forcings
# κ_I = 1
# κ_B = 1e2
# d = 500/4000*α
κ_I = 5.706e+00
κ_B = 2.535e+01
d = 3.526e-01*α
ν(x) = 1
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
b_surface_bc = SurfaceDirichletBC(b_surface)
conv_param = ConvectionParameterization(κᶜ=0.2/κ₀, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=sqrt(1e-3))
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)
@info @sprintf("Diffusion timescale: %.2e", (κ_B * ε^2 / μϱ)^-1)

# mesh
h = 2e-2
# mesh_name = @sprintf("channel_basin_no_flat_h%.2e_a%.2e", h, α)
mesh_name = @sprintf("channel_basin_flat_h%.2e_a%.2e", h, α)
if !isfile(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))
    # mesh_channel_basin_no_flat(h, α)
    mesh_channel_basin_flat(h, α)
end
mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

# # save κ
# writevtk(mesh.Ω, "$out_dir/data/kappa.vtu", cellfields=["kappa_v" => κᵥ, "kappa_h" => κₕ])

# FE data
u_diri_tags = ["bottom", "coastline", "surface"]
u_diri_vals = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
u_diri_masks = [(true, true, true), (true, true, true), (false, false, true)]
b_diri_tags = ["coastline", "surface"]
b_diri_vals = [b_surface, b_surface]
spaces = Spaces(mesh; u_diri_tags, u_diri_vals, u_diri_masks, b_diri_tags, b_diri_vals, b_order=1) 
fe_data = FEData(mesh, spaces)
display(fe_data.dofs)

# setup inversion toolkit
inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; itmax=1000)

# set timestepper
Δt = 1*86400/t₀
# t_stop = μϱ/ε^2/κ_B
t_stop = μϱ/ε^2/κ_I
timestepper = BDF1(; t_start=0, t_stop=t_stop, Δt=Δt, adaptive=true, CFL_factor=0.8)

# build evolution system
evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings, timestepper) 

# set up model
model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit, timestepper)

# set initial buoyancy
# set_b!(model, x -> b₀*x[3]/α + b_surface(x)*exp(x[3]/(α/4)))
set_state_from_file!(model, "/resnick/scratch/hppeters/sim051c/data/state_0000000000010800.jld2")
invert!(model)
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
@info @sprintf("Diffusion timescales: %.2e (κ_B), %.2e (κ_I)", μϱ/ε^2/κ_B, μϱ/ε^2/κ_I)
n_save = 100
n_plot = Inf
run!(model; n_save, n_plot)