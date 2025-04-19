using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

ENV["JULIA_DEBUG"] = nuPGCM
# ENV["JULIA_DEBUG"] = nothing

set_out_dir!(".")

# architecture and dimension
arch = GPU()
dim = 2

# params/funcs
ε = 1e-1
α = 1/2
μϱ = 1e0
N² = 1e0/α
Δt = 1e-4*μϱ/ε^2
params = Parameters(ε, α, μϱ, N², Δt)
f₀ = 1.0
β = 0.0
f(x) = f₀ + β*x[2]
H(x) = α*(1 - x[1]^2 - x[2]^2)
ν(x) = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
T = 5e-2*μϱ/ε^2
force_build_inversion_matrices = true
force_build_evolution_matrices = true

# mesh
h = 8e-3
mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))
# mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e_squash%d.msh", dim, h, 1, 1/α))

# build inversion matrices
A_inversion_fname = @sprintf("../matrices/A_inversion_%sD_%e_%e_%e_%e_%e.jld2", dim, h, ε, α, f₀, β)
if force_build_inversion_matrices
    @warn "You set `force_build_inversion_matrices` to `true`, building matrices..."
    A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; A_inversion_ofile=A_inversion_fname)
elseif !isfile(A_inversion_fname) 
    @warn "A_inversion file not found, generating..."
    A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; A_inversion_ofile=A_inversion_fname)
else
    file = jldopen(A_inversion_fname, "r")
    A_inversion = file["A_inversion"]
    close(file)
    B_inversion = nuPGCM.build_B_inversion(mesh, params)
end
@info "Approximate condition number of A_inversion: " cond(A_inversion, 1)
error()

# re-order dofs
A_inversion = A_inversion[mesh.dofs.p_inversion, mesh.dofs.p_inversion]
B_inversion = B_inversion[mesh.dofs.p_inversion, :]

# preconditioner
if typeof(arch) == CPU
    P_inversion = lu(A_inversion)
else
    P_inversion = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A_inversion, 1))))
end

# move to arch
A_inversion = on_architecture(arch, A_inversion)
B_inversion = on_architecture(arch, B_inversion)

# setup inversion toolkit
inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion)

# # quick inversion here:
# model = inversion_model(arch, params, mesh, inversion_toolkit)
# set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
# invert!(model)
# save_state(model, "$out_dir/data/state.jld2")

# build evolution matrices and test against saved matrices
θ = Δt/2 * ε^2 / μϱ
A_diff_fname = @sprintf("../matrices/A_diff_%sD_%e_%e_%e.jld2", dim, h, θ, α)
A_adv_fname = @sprintf("../matrices/A_adv_%sD_%e_%e.jld2", dim, h, α)
if !isfile(A_diff_fname) || !isfile(A_adv_fname) || force_build_evolution_matrices
    @warn "A_diff or A_adv file not found, generating..."
    A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ; 
                                        A_adv_ofile=A_adv_fname, A_diff_ofile=A_diff_fname)
else
    file = jldopen(A_adv_fname, "r")
    A_adv = file["A_adv"]
    close(file)
    file = jldopen(A_diff_fname, "r")
    A_diff = file["A_diff"]
    close(file)
    B_diff, b_diff = nuPGCM.build_B_diff_b_diff(mesh, params, κ)
end

# re-order dofs
A_adv  = A_adv[mesh.dofs.p_b, mesh.dofs.p_b]
A_diff = A_diff[mesh.dofs.p_b, mesh.dofs.p_b]
B_diff = B_diff[mesh.dofs.p_b, :]
b_diff = b_diff[mesh.dofs.p_b]

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
model = rest_state_model(arch, params, mesh, inversion_toolkit, evolution_toolkit)

# solve
run!(model, T; t_plot=T÷5)

println("Done.")