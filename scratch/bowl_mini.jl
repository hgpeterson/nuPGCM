using nuPGCM
using JLD2
using Printf

set_out_dir!(@__DIR__)

# params/funcs
arch = CPU()
dim = 3
ε = 2e-1
α = 1/2
μϱ = 1e1
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
n_steps = 500

# coarse mesh
h = 0.1
mesh = Mesh(joinpath(@__DIR__, @sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α)))

# FE data
u_diri = Dict("bottom"=>0, "coastline"=>0)
v_diri = Dict("bottom"=>0, "coastline"=>0)
w_diri = Dict("bottom"=>0, "coastline"=>0, "surface"=>0)
b_diri = Dict("surface"=>b_surface, "coastline"=>b_surface)
spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 
fe_data = FEData(mesh, spaces)

# setup inversion toolkit
inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings)

# build evolution system
evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

# put it all together in the `model` struct
model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

# solve
run!(model; n_steps)