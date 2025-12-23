using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

include(joinpath(@__DIR__, "../meshes/mesh_channel2D.jl"))  # for making channel2D mesh

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

set_out_dir!(joinpath(@__DIR__, "../sims/test"))

# architecture
arch = CPU()

# params
ε = sqrt(1e-1)
α = 1/8
μϱ = 1
N² = 0
Δt = 5e-4
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
function H(X)
    x = X[1]
    y = X[2]

    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = L_channel/4
    L_curve_channel = (L_channel - L_flat_channel)/2
    H = α*W

    # parabola that has a maximum of H at x_max and a 0 at x_zero
    parabola(x, x_max, x_zero) = H*(1 - ((x - x_max)/(x_zero - x_max))^2)

    if -L/2 ≤ y ≤ -L/2 + L_curve_channel
        return parabola(y, -L/2 + L_curve_channel, -L/2)
    elseif y ≤ -L/2 + 2L_curve_channel + L_flat_channel
        return H
    else
        throw(ArgumentError("y out of bounds"))
    end
end
params = Parameters(ε, α, μϱ, N², Δt, f, H)
display(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)

# forcings
ν(x) = 1
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
τ₀ = 1e-1
τˣ(x) = x[2] > -0.5 ? 0.0 : -τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
# b_surface(x) = 0
# b_surface(x) = x[2] > 0 ? 0.0 : -x[2]^2
b₀ = 10
# b_surface(x) = x[2] > 0 ? 0.0 : -b₀*(x[2] + 0.5)^2/0.5^2
# b_surface_bc = SurfaceDirichletBC(b_surface)
F₀ = 1
b_flux_surface(x) = x[2] > -0.5 ? 0.0 : -F₀*sin(2π*(x[2] + 1)/0.5)
b_surface_bc = SurfaceFluxBC(b_flux_surface)
# b_basin(x) = 0
b_basin(x) = b₀*x[3]/α
conv_param = ConvectionParameterization(κᶜ=1e3, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=1e-2)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)

# function setup_model()
    # mesh
    h = √2*α*ε/5
    mesh_name = @sprintf("channel2D_h%.2e_a%.2e", h, α)
    if !isfile(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))
        generate_channel_mesh_2D(h, α)
    end
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

    # FE data
    u_diri = Dict("bottom"=>0, "coastline"=>0, "basin bottom"=>0)
    v_diri = Dict("bottom"=>0, "coastline"=>0, "basin bottom"=>0, "basin top"=>0, "basin"=>0)
    w_diri = Dict("bottom"=>0, "coastline"=>0, "basin bottom"=>0, "basin top"=>0, "surface"=>0)
    # # SurfaceDirichletBC:
    # b_diri = Dict("coastline"=>b_surface, "surface"=>b_surface, "basin"=>b_basin, "basin bottom"=>b_basin, "basin top"=>b_basin)
    # # SurfaceFluxBC:
    # b_diri = Dict("basin"=>b_basin, "basin bottom"=>b_basin, "basin top"=>b_basin)
    # Sponge:
    b_diri = Dict()
    spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 
    fe_data = FEData(mesh, spaces)
    @info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; atol=1e-6, rtol=1e-6)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

#     return model
# end

# # set up model
# model = setup_model()

# # set initial buoyancy
# # set_b!(model, x -> 0)  # when N² is set
# set_b!(model, x -> b_basin(x))  # when N² = 0 
# # set_b!(model, x -> b_basin(x) + b_surface(x))  # when N² = 0 and SurfaceDirichletBC
# invert!(model)  # sync flow with buoyancy state
# save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))
# # i_step = 1400
# # set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", out_dir, i_step))
# # set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", joinpath(@__DIR__, "channel_2D/sim008"), i_step))

# # solve
# T = 10*μϱ/ε^2
# n_steps = Int(round(T / Δt))
# n_save = 10
# n_plot = Inf
# run!(model; n_steps, n_save, n_plot)
# # run!(model; n_steps, n_save, n_plot, i_step)

# println("Done.")