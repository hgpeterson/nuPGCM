using nuPGCM
using JLD2
using Printf

include(joinpath(@__DIR__, "../meshes/mesh_bowl2D.jl"))
include(joinpath(@__DIR__, "../meshes/mesh_bowl3D.jl"))

ENV["JULIA_DEBUG"] = nuPGCM
# ENV["JULIA_DEBUG"] = nothing

set_out_dir!(@__DIR__)

# params/funcs
arch = GPU()
dim = 3
ε = 1/2
α = 1/2
μϱ = 1
N² = 1/α
Δt = 1e-4*μϱ/(α*ε)^2
f₀ = 1
β = 0.5
f(x) = f₀ + β*x[2]
H(x) = α*(1 - x[1]^2 - x[2]^2)
params = Parameters(ε, α, μϱ, N², Δt, f, H)
ν = 1
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
τˣ(x) = 0
τʸ(x) = 0
b_surface(x) = 0
b_surface_bc = SurfaceDirichletBC(b_surface)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
T = 0.1*μϱ/ε^2
n_steps = T ÷ Δt

# mesh
h = 0.2*α
mesh_file = joinpath(@__DIR__, @sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))
if !isfile(mesh_file)
    if dim == 2
        generate_bowl_mesh_2D(h, α)
    elseif dim == 3
        generate_bowl_mesh_3D(h, α)
    end
end
mesh = Mesh(mesh_file)

# FE data
u_diri = Dict("bottom"=>0, "coastline"=>0)
v_diri = Dict("bottom"=>0, "coastline"=>0)
w_diri = Dict("bottom"=>0, "coastline"=>0, "surface"=>0)
b_diri = Dict("surface"=>b_surface, "coastline"=>b_surface)
spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 
fe_data = FEData(mesh, spaces)
@info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

# setup inversion toolkit
inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings)

# test inversion
model = Model(arch, params, forcings, fe_data, inversion_toolkit)
# set_b!(model, x->x[3]/α)
set_state_from_file!(model.state, @sprintf("%s/data/state_a%e.jld2", out_dir, α))
invert!(model)

# # build evolution system
# evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

# # put it all together in the `model` struct
# model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

# # initial condition
# set_b!(model, x->0)
# # invert!(model)  # sync flow with buoyancy state
# # save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# # solve
# n_save = n_steps
# run!(model; n_steps, n_save, advection=false)

# mv(@sprintf("%s/data/state_%016d.vtu", out_dir, n_steps), @sprintf("%s/data/state_a%e.vtu", out_dir, α), force=true)
# mv(@sprintf("%s/da$ta/state_%016d.jld2", out_dir, n_steps), @sprintf("%s/data/state_a%e.jld2", out_dir, α), force=true)