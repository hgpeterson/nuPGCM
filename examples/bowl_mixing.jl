# # Mixing-driven circulation in a bowl-shaped basin
#
# This example illustrates the basic usage of the $\nu$PGCM by simulating a 
# mixing-driven circulation in a bowl-shaped basin. The typical workflow is:
#
# * Set your `Parameters` and `Forcings`,
# * Load a `Mesh`,
# * Define the finite element `Spaces`, including all Dirichlet boundary conditions,
# * Build the linear systems for the inversion and evolution equations,
# * Construct a `Model`, and
# * `run!`
#
# We will start by importing packages and setting the output directory for snapshot files.

using nuPGCM
using Printf

set_out_dir!(joinpath(@__DIR__, ""))

# You can choose to run the model on either a `CPU()` or `GPU()`. When you 
# run on the CPU, the model will try to factorize matrices if they are small 
# enough. On the GPU (or for large problems on the CPU), the model uses 
# iterative solvers. For simplicity here, let's just use a CPU.

arch = CPU()

# ## Setting `Parameters` and `Forcings`

# Now we will define our parameters. See the **Model Formulation** docs
# for more details. Our bowl-shaped basin has a simple depth function of
# $H(x, y) = \alpha (1 - x^2 - y^2)$.

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

# Next, we define the forcings. For this simple example, we'll be applying 
# bottom-enhanced mixing and no wind stress.

ν = 1  # viscosity (can be a function of x)
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α)) # horizontal diffusivity
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α)) # vertical diffusivity
τˣ(x) = 0  # zonal wind stress
τʸ(x) = 0  # meridional wind stress

# You can choose whether the surface buoyancy forcing is a flux boundary 
# condition (with `SurfaceFluxBC`) or a dirichlet boundary condition 
# (with `SurfaceDirichletBC`). Here we'll just set `b = 0` at the surface.

b_surface(x) = 0  # surface buoyancy boundary condition
b_surface_bc = SurfaceDirichletBC(b_surface)
## flux syntax:
## b_flux_surface(x) = 0
## b_surface_bc = SurfaceFluxBC(b_flux_surface)

forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)

# Notice that `forcings.conv_param` and `forcings.eddy_param` are by default
# set to `off`. These (somewhat experimental) new features can be defined 
# like so:

conv_param = ConvectionParameterization(κᶜ=1e3, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=1e-2)
Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)

# But we'll leave them off here.

# ## Loading a `Mesh`

# Now we load our mesh. For some examples of how to generate a mesh with [Gmsh](https://gmsh.info/), see 
# [`mesh_bowl2D.jl`](https://github.com/hgpeterson/nuPGCM/blob/main/meshes/mesh_bowl2D.jl),
# [`mesh_bowl3D.jl`](https://github.com/hgpeterson/nuPGCM/blob/main/meshes/mesh_bowl3D.jl), and others in
# the `meshes/` directory of the repository.

h = 8e-2
dim = 3
mesh_file = joinpath(@__DIR__, @sprintf("../../../meshes/bowl%dD_%e_%e.msh", dim, h, α))  # for Literate.jl
## mesh_file = joinpath(@__DIR__, @sprintf("../meshes/bowl%dD_%e_%e.msh", dim, h, α))  # for running from `examples/`
mesh = Mesh(mesh_file)

# !!! note 
#     You may have noticed that the `Mesh` fields are 
#     [`Gridap`](https://github.com/gridap/Gridap.jl) types. Under the hood, 
#     `nuPGCM` uses `Gridap` to compute all the finite element integrals. 
#     When we define the `Model` later, the `State` will contain `Gridap` 
#     `FEFunction`s.

# # Define `Spaces`

# As described on the [Numerical Approach](../model_formulation/numerical_approach.md) page, the $\nu$PGCM
# uses finite elements under the hood to solve the PG equations. Now that we have
# an unstructured mesh set up, we just need to define our Dirichlet boundary
# conditions to  be able to set up our finite element spaces. 

# In Gmsh, one can assign labels to parts of the mesh by defining "physical groups."
# For the mesh we're using here, there are three physical groups defined: the `"bottom"`,
# `"coastline"`, and `"surface"`. To tell the $\nu$PGCM where to apply Dirichlet boundary
# conditions, we define dictionaries for $u$, $v$, $w$, and $b$ using these labels.

u_diri = Dict("bottom"=>0, "coastline"=>0)
v_diri = Dict("bottom"=>0, "coastline"=>0)
w_diri = Dict("bottom"=>0, "coastline"=>0, "surface"=>0)
b_diri = Dict("surface"=>b_surface, "coastline"=>b_surface) 
## b_diri = Dict()  # use this if b_surface_bc is a SurfaceFluxBC
spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 

# We have enforced the $u = v = w = 0$ on the `"bottom"` and `"coastline"`, $w = 0$ at 
# the `"surface"`, and $b = 0$ at the `"surface"` and on the `"coastline"`. All other 
# boundaries are treated "naturally," i.e., the flux across them is zero.

# !!! warning "Be careful about defining your physical groups!"
#     Make sure that when you create your `.msh` files you carefully check that
#     your physical groupd definitions properly account for every entity in the
#     mesh! For some examples, see the 
#     [`meshes/`](https://github.com/hgpeterson/nuPGCM/tree/main/meshes) 
#     folder in the `nuPGCM` repository.

# Our `mesh` and `spaces` are then passed to the `FEData` constructor, which 
# keeps track of all the finite element data needed to compute build our 
# linear systems.

fe_data = FEData(mesh, spaces)

# Behind the scenes, this created a `DoFHandler`, which you can use to print
# out how many degrees of freedom the inversion system will have.

@info "Inversion DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

# # Build linear systems

# We now have everything we need to build the linear systems that represent
# the discretized PG equations on this mesh. First, we build the inversion
# system alone, as it does not have a time-dependent piece:

inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; atol=1e-6, rtol=1e-6)

# The `atol` and `rtol` kwargs are the absolute and relative tolerances, 
# respectively, used when solving the system iteratively. Make them smaller if
# you want higher accuracy at the expense of more inversion iterations. Since
# we are on a CPU and our mesh is quite coarse in this example, however, the 
# model will actually just do a direct solve, so these tolerances are unused. 

# At this point, we can already compute the flow field if we know what the 
# buoyancy field is. To do this, just create a `Model` without an evolution
# piece, set the buoyancy field, and invert:
 
model = Model(arch, params, forcings, fe_data, inversion_toolkit)
set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
invert!(model)
save_state(model, "$out_dir/data/state.jld2")
save_vtk(model, ofile="$out_dir/data/state.vtu")
 
# Here we want to see how the flow and buoyancy evolve in time, however, so 
# we need to build the evolution system.

evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings)

# # Construct `Model`

# Now we put everything together in the `Model` type.

model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

# We can set the intial condition to be whatever we want with `set_b!(model, foo)`
# where `foo` is a function of `x`. By default, the buoyancy is always set to 0
# initially. Since we set `N²` to `1/α`, this means that our buoyancy field starts
# out as `x[3]/α`―a constant stratification. If you had `N² = 0`, then you would
# need to do `set_b!(model, x -> x[3]/α)` to get the same effect. The benefit of 
# having a nonzero background startification is that the inversion tends to be more
# accurate for smaller variations in buoyancy.

# To start with, let's sync up the flow with whatever initial condition we chose
# and save a `.vtu` file.
 
invert!(model) # sync flow with initial condition 
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# # `run!`

# Finally, it's time to run the model! `run!` just needs to know how many steps
# to take `n_steps`, how often to make save files `n_save` (default `Inf`),
# and how often to make plots `n_plot` (default `Inf`). If you want to turn
# advection off and only diffuse buoyancy, set `advection=false`. If you are
# starting from a save file, you can also set `i_step` to something other than `1`.

T = 0.1*μϱ/ε^2  # simulation time
n_steps = Int(round(T / Δt))
n_save = n_steps ÷ 100
run!(model; n_steps, n_save)