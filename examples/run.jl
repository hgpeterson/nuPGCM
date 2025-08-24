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
N² = 1/α     # background stratification (if you want `b` to be a perturbation from N²z)
Δt = 1e-3  # time step
params = Parameters(ε, α, μϱ, N², Δt)
T = 0.1*μϱ/ε^2  # simulation time
f₀ = 1.0
β = 0.5
f(x) = f₀ + β*x[2]  # Coriolis parameter
H(x) = α*(1 - x[1]^2 - x[2]^2)  # bathymetry
ν(x) = 1  # viscosity
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))  # diffusivity
τˣ(x) = 0  # zonal wind stress
τʸ(x) = 0  # meridional wind stress
b₀(x) = 0  # surface buoyancy boundary condition
force_build_inversion = false
force_build_evolution = true

# mesh
h = 8e-2
dim = 3
mesh_name = @sprintf("bowl%dD_%e_%e", dim, h, α)
mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

# FE data
spaces = Spaces(mesh, b₀)
fe_data = FEData(mesh, spaces)
@info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

# build inversion matrices
A_inversion_fname = joinpath(@__DIR__, "../matrices/A_inversion_$mesh_name.jld2")
if force_build_inversion
    @warn "You set `force_build_inversion` to `true`, building matrices..."
    A_inversion, B_inversion, b_inversion = build_inversion_matrices(fe_data, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
elseif !isfile(A_inversion_fname) 
    @warn "A_inversion file not found, generating..."
    A_inversion, B_inversion, b_inversion = build_inversion_matrices(fe_data, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
else
    file = jldopen(A_inversion_fname, "r")
    A_inversion = file["A_inversion"]
    close(file)
    B_inversion = nuPGCM.build_B_inversion(fe_data, params)
    b_inversion = nuPGCM.build_b_inversion(fe_data, params, τˣ, τʸ)
end

# re-order dofs
A_inversion = A_inversion[fe_data.dofs.p_inversion, fe_data.dofs.p_inversion]
B_inversion = B_inversion[fe_data.dofs.p_inversion, :]
b_inversion = b_inversion[fe_data.dofs.p_inversion]

# preconditioner
if typeof(arch) == CPU
    @time "lu(A_inversion)" P_inversion = lu(A_inversion)
else
    P_inversion = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A_inversion, 1))))
end

# move to arch
A_inversion = on_architecture(arch, A_inversion)
B_inversion = on_architecture(arch, B_inversion)
b_inversion = on_architecture(arch, b_inversion)

# setup inversion toolkit
inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion, b_inversion; atol=1e-6, rtol=1e-6)

# # quick inversion here:
# model = inversion_model(arch, params, mesh, inversion_toolkit)
# set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
# invert!(model)
# save_state(model, "$out_dir/data/state.jld2")

# build evolution matrices (or load them if `force_build` is false and file exists)
A_adv, A_diff, B_diff, b_diff = build_evolution_system(fe_data, params, κ; 
                                    force_build=force_build_evolution,
                                    filename=joinpath(@__DIR__, "../matrices/evolution_$mesh_name.jld2"))

# re-order dofs
A_adv  =  A_adv[fe_data.dofs.p_b, fe_data.dofs.p_b]
A_diff = A_diff[fe_data.dofs.p_b, fe_data.dofs.p_b]
B_diff = B_diff[fe_data.dofs.p_b, :]
b_diff = b_diff[fe_data.dofs.p_b]

# preconditioners
if typeof(arch) == CPU 
    P_diff = lu(A_diff)
    P_adv  = lu(A_adv)
else
    P_diff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_diff))))
    P_adv  = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_adv))))
end

# move to arch
A_adv  = on_architecture(arch, A_adv)
A_diff = on_architecture(arch, A_diff)
B_diff = on_architecture(arch, B_diff)
b_diff = on_architecture(arch, b_diff)

# setup evolution toolkit
evolution_toolkit = EvolutionToolkit(A_adv, P_adv, A_diff, P_diff, B_diff, b_diff)

# put it all together in the `model` struct
model = rest_state_model(arch, params, fe_data, inversion_toolkit, evolution_toolkit)

# set initial buoyancy
set_b!(model, x->b₀(x))
invert!(model) # sync flow with initial condition 
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
n_steps = Int(round(T / Δt))
n_save = n_steps ÷ 100
run!(model; n_steps, n_save)

println("Done.")