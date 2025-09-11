using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

ENV["JULIA_DEBUG"] = nuPGCM
# ENV["JULIA_DEBUG"] = nothing

set_out_dir!(joinpath(@__DIR__, ""))

# architecture
arch = GPU()

# params
ε = 1e-1
α = 1/2
μϱ = 1
N² = 0
Δt = 1e-3
κᶜ = 100
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
H_basin(x) = α*(x[1]*(1 - x[1]))/(0.5*0.5)
H_channel(x) = -α*((x[2] + 1)*(x[2] + 0.5))/(0.25*0.25)
H(x) = x[2] > -0.75 ? max(H_channel(x), H_basin(x)) : H_channel(x)
params = Parameters(ε, α, μϱ, N², Δt, κᶜ, f, H)
display(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)

# forcings
ν(x) = 1
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
τˣ(x) = x[2] > -0.5 ? 0.0 : -1e-4*(x[2] + 1)*(x[2] + 0.5)/(1 + 0.5)^2
τʸ(x) = 0
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b₀)

function setup_model()
    # mesh
    mesh_name = "channel_basin"
    # h = 4e-2
    # mesh_name = @sprintf("channel_basin_%.1e_%.1e", h, α)
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

    # FE data
    spaces = Spaces(mesh, b₀)
    fe_data = FEData(mesh, spaces, forcings)
    @info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; atol=1e-6, rtol=1e-6)

    # # quick inversion here:
    # model = inversion_model(arch, params, mesh, inversion_toolkit)
    # # set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
    # # set_b!(model, x -> 0.1*exp(-((x[1] - 0.5)^2 + (x[2] + 0.5)^2 + (x[3] + α/2)^2)/2/0.1^2))
    # # set_b!(model, x -> 0.1*exp(-((x[1] - 0.3)^2 + (x[2] + 0.75)^2 + (x[3] + α/2)^2)/2/0.1^2))
    # # set_b!(model, x -> 0.1*exp(-((x[1] - 0.3)^2 + x[2]^2 + (x[3] + α/2)^2)/2/0.1^2))
    # # set_b!(model, x -> 0.1*exp(-(x[1]^2 + (x[3] + 0.5)^2)/2/0.1^2) + 
    # #                    0.1*exp(-((x[1] - 1)^2 + (x[3] + 0.5)^2)/2/0.1^2))
    # # set_b!(model, x -> 1/α*x[3])
    # invert!(model)
    # save_state(model, "$out_dir/data/state.jld2")

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = rest_state_model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    return model
end

# set up model
model = setup_model()

# # set initial buoyancy
# set_b!(model, x->b₀(x) + x[3]/α)
# nuPGCM.update_κᵥ!(model, model.state.b)  # reset κᵥ
# invert!(model) # sync flow with buoyancy state
# save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# # solve
# T = μϱ/ε^2
# n_steps = Int(round(T / Δt))
# # n_save = n_steps ÷ 100
# n_save = 100
# n_plot = Inf
# run!(model; n_steps, n_save, n_plot)

println("Done.")