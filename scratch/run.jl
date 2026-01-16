using nuPGCM
using JLD2
using Printf

# for making mesh
# include(joinpath(@__DIR__, "../meshes/channel_basin.jl"))  
include(joinpath(@__DIR__, "../meshes/channel_basin_flat.jl"))  

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

set_out_dir!(joinpath(@__DIR__, ""))

# architecture
arch = GPU()

# params

# Ω = 2π/86400  # s⁻¹
# a = 6.371e6  # m
# β = 2Ω/a  # m⁻¹ s⁻¹
# L = 2π*a*60/360  # m
# f₀ = β*L  # s⁻¹
# H₀ = 4e3  # m
# κ₀ = 3e-4  # m² s⁻¹
# Kₑ = 1000  # m² s⁻¹
# N₀ = 1e-3  # s⁻¹
# ρ₀ = 1035  # kg m⁻³
# α_T = 2e-4  # °C⁻¹
# g = 9.81  # m s⁻²
# ν₀ = Kₑ*f₀^2/N₀^2  # m² s⁻¹
# τ₀ = ρ₀*N₀^2*H₀^3/L  # N m⁻²
# b₀ = g*α_T*30/(N₀^2*H₀)

ν₀ = 10  # m² s⁻¹
κ₀ = 1e-3  # m² s⁻¹
f₀ = 1e-4  # s⁻¹
N₀ = 1e-3  # s⁻¹
H₀ = 1e3  # m
L = 1e6  # m
τ₀ = 1  # N m⁻²
b₀ = 10

ε = sqrt(ν₀/f₀/H₀^2)
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2

t₀ = 1/f₀/ϱ  # s
@info "scales" b₀ ν₀ τ₀ t₀

μϱ = μ*ϱ
α = 1/4
N² = 0
# Δt = 10*86400/t₀
Δt = 2e-3
f(x) = x[2]
H(x) = α
# curved_southern_bdy = true
# function H(xyz)
#     x = xyz[1]
#     y = xyz[2]

#     L = 2
#     W = 1
#     L_channel = L/4
#     L_flat_channel = L_channel/4 # length of flat part of channel
#     L_curve_channel = (L_channel - L_flat_channel)/2 # length of each curved part of channel
#     W_flat_basin = W/2 # width of flat part of basin
#     W_curve_basin = (W - W_flat_basin)/2 # width of each curved part of basin
#     L_curve_basin = W_curve_basin # length of curved end of basin
#     H = α*W

#     # parabola that has a maximum of H at x_max and a 0 at x_zero
#     parabola(x, x_max, x_zero) = H*(1 - ((x - x_max)/(x_zero - x_max))^2)

#     function H_basin(x)
#         if 0 ≤ x ≤ W_curve_basin
#             return parabola(x, W_curve_basin, 0)
#         elseif x ≤ W_curve_basin + W_flat_basin
#             return H
#         elseif x ≤ W
#             return parabola(x, W_curve_basin + W_flat_basin, W)
#         else
#             throw(ArgumentError("x out of bounds"))
#         end
#     end

#     if -L/2 ≤ y ≤ -L/2 + L_curve_channel
#         if curved_southern_bdy
#             return parabola(y, -L/2 + L_curve_channel, -L/2)
#         else
#             return H
#         end
#     elseif y ≤ -L/2 + L_curve_channel + L_flat_channel
#         return H
#     elseif y ≤ -L/2 + L_channel
#         H_channel = parabola(y, -L/2 + L_curve_channel + L_flat_channel, -L/2 + L_channel)
#         return max(H_channel, H_basin(x))
#     elseif y ≤ L/2 - L_curve_basin
#         return H_basin(x)
#     elseif y ≤ L/2
#         if 0 ≤ x ≤ W_curve_basin
#             x₀ = W_curve_basin
#             y₀ = L/2 - L_curve_basin
#             r = √( (x - x₀)^2 + (y - y₀)^2 )
#             return parabola(r, 0, W_curve_basin)
#         elseif W_curve_basin ≤ x ≤ W_curve_basin + W_flat_basin
#             return parabola(y, L/2 - L_curve_basin, L/2)
#         elseif x ≤ W
#             x₀ = W_curve_basin + W_flat_basin
#             y₀ = L/2 - L_curve_basin
#             r = √( (x - x₀)^2 + (y - y₀)^2 )
#             return parabola(r, 0, W_curve_basin)
#         else
#             throw(ArgumentError("x out of bounds"))
#         end
#     else
#         throw(ArgumentError("y out of bounds"))
#     end
# end
params = Parameters(ε, α, μϱ, N², Δt, f, H)
display(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)

# forcings
ν(x) = 1
κₕ(x) = 1
κᵥ(x) = 1
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.1/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
# τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
b_surface_bc = SurfaceDirichletBC(b_surface)
# F₀ = 1
# b_flux_surface(x) = x[2] > -0.5 ? 0.0 : -F₀*sin(2π*(x[2] + 1)/0.5)
# b_surface_bc = SurfaceFluxBC(b_flux_surface)
conv_param = ConvectionParameterization(κᶜ=5e2, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=1e-3)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)

function setup_model()
    # mesh
    # h = √2*α*ε
    h = 0.1
    # if curved_southern_bdy
    #     mesh_name = @sprintf("channel_basin_h%.2e_a%.2e", h, α)
    # else
    #     mesh_name = @sprintf("channel_basin_h%.2e_a%.2e_vert_sb", h, α)
    # end   
    mesh_name = @sprintf("channel_basin_flat_h%.2e_a%.2e", h, α)
    if !isfile(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))
        # mesh_channel_basin(h, α; curved_southern_bdy)
        mesh_channel_basin_flat(h, α)
    end
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

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
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; itmax=10_000, atol=1e-6, rtol=1e-6)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings; order=1) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    return model
end

# set up model
model = setup_model()
display(model)

# set initial buoyancy
# set_b!(model, x -> 0)  # when N² is set
# set_b!(model, x -> b₀*x[3]/α)  # when N² = 0 
set_b!(model, x -> b₀*x[3]/α + b_surface(x)*exp(x[3]/(α/4)))  # when N² = 0 and SurfaceDirichletBC
# set_b!(model, x -> b₀*x[3]/α + b_surface(x))  # when N² = 0 and SurfaceDirichletBC
invert!(model)  # sync flow with buoyancy state
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))
# i_step = 660
# set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", out_dir, i_step))
# set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", joinpath(@__DIR__, "channel_2D/sim008"), i_step))

# solve
T = 2*μϱ/ε^2
n_steps = Int(round(T / Δt))
n_save = 100
n_plot = Inf
run!(model; n_steps, n_save, n_plot)
# run!(model; n_steps, n_save, n_plot, i_step)

println("Done.")
