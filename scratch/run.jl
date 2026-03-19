using nuPGCM
using Gridap
using JLD2
using Printf

nuPGCM_dir = "/resnick/groups/oceanphysics/henry/nuPGCM"

# for making mesh
include("$nuPGCM_dir/meshes/channel_basin.jl")

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

# set_out_dir!(joinpath(@__DIR__, "adaptive_timestep"))
set_out_dir!("/resnick/scratch/hppeters/sim050")

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
Δt = 3600/t₀
f(x) = x[2]
function H(xyz)
    x = xyz[1]
    y = xyz[2]

    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = L_channel/4 # length of flat part of channel
    L_curve_channel = (L_channel - L_flat_channel)/2 # length of each curved part of channel
    W_flat_basin = W/2 # width of flat part of basin
    W_curve_basin = (W - W_flat_basin)/2 # width of each curved part of basin
    L_curve_basin = W_curve_basin # length of curved end of basin
    H = α*W

    # parabola that has a maximum of H at x_max and a 0 at x_zero
    parabola(x, x_max, x_zero) = H*(1 - ((x - x_max)/(x_zero - x_max))^2)

    function H_basin(x)
        if 0 ≤ x ≤ W_curve_basin
            return parabola(x, W_curve_basin, 0)
        elseif x ≤ W_curve_basin + W_flat_basin
            return H
        elseif x ≤ W
            return parabola(x, W_curve_basin + W_flat_basin, W)
        else
            throw(ArgumentError("x out of bounds"))
        end
    end

    if -L/2 ≤ y ≤ -L/2 + L_curve_channel
        return H
    elseif y ≤ -L/2 + L_curve_channel + L_flat_channel
        return H
    elseif y ≤ -L/2 + L_channel
        H_channel = parabola(y, -L/2 + L_curve_channel + L_flat_channel, -L/2 + L_channel)
        return max(H_channel, H_basin(x))
    elseif y ≤ L/2 - L_curve_basin
        return H_basin(x)
    elseif y ≤ L/2
        if 0 ≤ x ≤ W_curve_basin
            x₀ = W_curve_basin
            y₀ = L/2 - L_curve_basin
            r = √( (x - x₀)^2 + (y - y₀)^2 )
            return parabola(r, 0, W_curve_basin)
        elseif W_curve_basin ≤ x ≤ W_curve_basin + W_flat_basin
            return parabola(y, L/2 - L_curve_basin, L/2)
        elseif x ≤ W
            x₀ = W_curve_basin + W_flat_basin
            y₀ = L/2 - L_curve_basin
            r = √( (x - x₀)^2 + (y - y₀)^2 )
            return parabola(r, 0, W_curve_basin)
        else
            throw(ArgumentError("x out of bounds"))
        end
    else
        throw(ArgumentError("y out of bounds"))
    end
end
params = Parameters(ε, α, μϱ, N², Δt, f, H)
display(params)

# forcings
ν(x) = 1
κ_B = 1e2
κ_I = 1
d = 500/4000 * α
# d = 500/2000 * α  # 2x larger decay scale
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
# κₕ(x) = κ_I
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
b_surface_bc = SurfaceDirichletBC(b_surface)
conv_param = ConvectionParameterization(κᶜ=0.2/κ₀, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=1e-3)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)

function setup_model()
    # mesh
    h = 2e-2
    refinement_factor = 2
    if refinement_factor === nothing
        mesh_name = @sprintf("channel_basin_h%.2e_a%.2e", h, α)
    else
        mesh_name = @sprintf("channel_basin_h%.2e_a%.2e_r%d", h, α, refinement_factor)
    end   
    if !isfile("$nuPGCM_dir/meshes/$mesh_name.msh")
        mesh_channel_basin(h, α; refinement_factor)
    end
    mesh = Mesh("$nuPGCM_dir/meshes/$mesh_name.msh")

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
set_b!(model, x -> b_surface(x))
invert!(model)
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
t_stop = μϱ/ε^2/κ_B
@info @sprintf("Diffusion timescale: %.2e (κ_B), %.2e (κ_I)", μϱ/ε^2/κ_B, μϱ/ε^2/κ_I)
n_save = 100
n_plot = Inf
run!(model; t_stop, n_save, n_plot)

println("Done.")
