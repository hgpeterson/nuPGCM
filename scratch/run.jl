using nuPGCM
using CUDA
using JLD2
using Printf

ENV["JULIA_DEBUG"] = nuPGCM
# ENV["JULIA_DEBUG"] = nothing
ENABLE_TIMING[] = true

PROJ_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM-ferrite"
# SIMS_PATH = "/resnick/scratch/hppeters"
SIMS_PATH = @__DIR__

set_out_dir!(joinpath(SIMS_PATH, "channel_basin_000"))

# geom = :tub
geom = :box

# for making mesh
if geom == :tub
    include(joinpath(PROJ_PATH, "meshes/channel_basin_no_flat_round_end.jl"))
elseif geom == :box
    include(joinpath(PROJ_PATH, "meshes/channel_basin_flat.jl"))
end

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
# α = 1/8
N² = 0.0
f(x) = x[2]
function H((x, y, z))
    if geom == :box
        return α
    end
    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = 5L_channel/8
    H = α*W

    parabola(x, x_max, x_zero) = H*(1 - ((x - x_max)/(x_zero - x_max))^2)

    function H_basin(x)
        if 0 ≤ x ≤ W
            return parabola(x, W/2, 0)
        else
            throw(ArgumentError("x out of bounds"))
        end
    end

    if -L/2 ≤ y ≤ -L/2 + L_flat_channel
        return H
    elseif y ≤ -L/2 + L_channel
        H_channel = parabola(y, -L/2 + L_flat_channel, -L/2 + L_channel)
        return max(H_channel, H_basin(x))
    elseif y ≤ L/2 - W/2
        return H_basin(x)
    elseif y ≤ L/2
        r = √( (x - W/2)^2 + (y - (L/2 - W/2))^2 )
        if r > W/2
            if r - W/2 < 1e-1
                return 0
            else
                throw(ArgumentError("(x, y) out of bounds"))
            end
        else
            return parabola(r, 0, W/2)
        end
    else
        throw(ArgumentError("y out of bounds"))
    end
end
params = Parameters(; ε, α, μϱ, N², f, H)
display(params)

# resolution
h = 4e-2
# h = 2e-2
# h = 1e-2

# forcings
if geom == :tub
    κ_I = 1
    κ_B = 1e2
    d = 500/4000*α
elseif geom == :box
    κ_I = 5.706e+00
    κ_B = 2.535e+01
    d = 3.526e-01*α
    # κ_I = 3.752e+00  # wall
    # κ_B = 2.834e+01
    # d = 3.881e-01*α
end
ν(x) = 1.0
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0.0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
b_surface_bc = SurfaceDirichletBC(b_surface)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
display(forcings)
@info @sprintf("Diffusion timescale: %.2e", (κ_B * ε^2 / μϱ)^-1)

# mesh
if geom == :tub
    mesh_name = @sprintf("channel_basin_no_flat_h%.2e_a%.2e", h, α)
elseif geom == :box
    mesh_name = @sprintf("channel_basin_flat_h%.2e_a%.2e", h, α)
end
mesh_file = joinpath(PROJ_PATH, "meshes/$mesh_name.msh")
if !isfile(mesh_file)
    if geom == :tub
        mesh_channel_basin_no_flat(h, α)
    elseif geom == :box
        mesh_channel_basin_flat(h, α)
    end
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
inv_tk = InversionToolkit(arch, fe_data, params, forcings; itmax=1000)

# set timestepper
Δt = 1*86400/t₀
t_stop = μϱ/ε^2/κ_I
ts = BDF1(; t_start=0.0, t_stop, Δt, adaptive=true, CFL_factor=0.8)

# build evolution toolkit
evo_tk = EvolutionToolkit(arch, fe_data, params, forcings, ts)

# set up model
model = Model(arch, params, forcings, fe_data, inv_tk, evo_tk, ts)

# set initial buoyancy
set_b!(model, x -> -b₀ + (b_surface(x) + b₀)*exp(x[3]/(α/4)))
# i_start = 11700
# set_state_from_file!(model, joinpath(out_dir, @sprintf("data/state_%016d.jld2", i_start)))

# solve
@info @sprintf("Diffusion timescales: %.2e (κ_B), %.2e (κ_I)", μϱ/ε^2/κ_B, μϱ/ε^2/κ_I)
n_save = 100
run!(model; n_save)
# run!(model; i_start, n_save)