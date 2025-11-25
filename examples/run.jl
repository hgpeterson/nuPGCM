using nuPGCM
using JLD2
using LinearAlgebra
using Printf

set_out_dir!(joinpath(@__DIR__, ""))

# architecture
arch = CPU()

# params
ε = 2e-1   # Ekman number
α = 1/2    # aspect ratio
μϱ = 1     # Prandtl times Burger number
N² = 1/α   # background stratification (if you want `b` to be a perturbation from N²z)
Δt = 1e-3  # time step
f₀ = 1.0
β = 0.5
f(x) = f₀ + β*x[2]  # Coriolis parameter
H(x) = α*(1 - x[1]^2 - x[2]^2)  # bathymetry
params = Parameters(ε, α, μϱ, N², Δt, f, H)

# forcings
ν = 1  # viscosity (can be a function of x)
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α)) # horizontal diffusivity
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α)) # vertical diffusivity
τˣ(x) = 0  # zonal wind stress
τʸ(x) = 0  # meridional wind stress

# dirichlet surface boundary condition for buoyancy
b_surface(x) = 0  # surface buoyancy boundary condition
b_surface_bc = SurfaceDirichletBC(b_surface)

# # flux surface boundary condition for buoyancy
# b_flux_surface(x) = 0
# b_surface_bc = SurfaceFluxBC(b_flux_surface)

forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)

# # somewhat experimental: convection and eddy parameterizations:
# conv_param = ConvectionParameterization(κᶜ=1e3, N²min=1e-3)
# eddy_param = EddyParameterization(f=f, N²min=1e-2)
# forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)

# mesh (see mesh_bowl2D.jl and mesh_bowl3D.jl for examples of how to generate a mesh with Gmsh)
h = 8e-2
dim = 3
mesh_name = @sprintf("bowl%dD_%e_%e", dim, h, α)
mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

# dirichlet BCs
u_diri = Dict("bottom"=>0, "coastline"=>0)
v_diri = Dict("bottom"=>0, "coastline"=>0)
w_diri = Dict("bottom"=>0, "coastline"=>0, "surface"=>0)
b_diri = Dict("surface"=>b_surface, "coastline"=>b_surface) 
# b_diri = Dict()  # use this if b_surface_bc is a SurfaceFluxBC
spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 
fe_data = FEData(mesh, spaces)
@info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

# build inversion system
inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; atol=1e-6, rtol=1e-6)

# # if all you want is a quick inversion for the flow given b, do this:
# model = Model(arch, params, forcings, fe_data, inversion_toolkit)
# set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
# invert!(model)
# save_state(model, "$out_dir/data/state.jld2")
# save_vtk(model, ofile="$out_dir/data/state.vtu")

# build evolution system
evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

# put it all together in the `model` struct
model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

# set initial buoyancy (default 0)
# set_b!(model, x -> x[3]/α)  # use this if N² = 0
invert!(model) # sync flow with initial condition 
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
T = 0.1*μϱ/ε^2  # simulation time
n_steps = Int(round(T / Δt))
n_save = n_steps ÷ 100
run!(model; n_steps, n_save)

println("Done.")