using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

set_out_dir!(".")

# architecture
arch = CPU()

# params/funcs
ε = 1e-1
α = 0.5
μϱ = 1e0
N² = 1e0/α
Δt = 1e-2
params = Parameters(ε, α, μϱ, N², Δt)
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
# H(x) = α*(1 - x[1]^2 - x[2]^2)
include("get_H.jl")
ν(x) = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
# κ(x) = 1
y0 = -0.5
τx(x) = x[2] > y0 ? 0.0 : -(x[2] + 1)*(x[2] - y0)/(1 - y0)^2
τy(x) = 0
T = 1e1
force_build_inversion_matrices = false
force_build_evolution_matrices = true

# mesh
# mesh_name = "periodic_square"
# mesh_name = "channel"
mesh_name = "channel_basin_cart"
mesh = Mesh("../meshes/$mesh_name.msh")
@info "DOFs: $(mesh.dofs.nu + mesh.dofs.nv + mesh.dofs.nw + mesh.dofs.np)" 
p, t = nuPGCM.get_p_t(mesh.model)
hs = [norm(p[t[k, i], :] - p[t[k, mod1(i+1, size(t, 2))], :]) for k in axes(t, 1), i in axes(t, 2)]
h = sum(hs) / length(hs)
@info "Mesh size: $h"
dim = size(t, 2) - 1
@info "Mesh dimension: $dim"

# build inversion matrices
A_inversion_fname = "../matrices/A_inversion_$mesh_name.jld2"
if force_build_inversion_matrices
    @warn "You set `force_build_inversion_matrices` to `true`, building matrices..."
    A_inversion, B_inversion, b_inversion = build_inversion_matrices(mesh, params, f, ν, τx, τy; A_inversion_ofile=A_inversion_fname)
elseif !isfile(A_inversion_fname) 
    @warn "A_inversion file not found, generating..."
    A_inversion, B_inversion, b_inversion = build_inversion_matrices(mesh, params, f, ν, τx, τy; A_inversion_ofile=A_inversion_fname)
else
    file = jldopen(A_inversion_fname, "r")
    A_inversion = file["A_inversion"]
    close(file)
    B_inversion = nuPGCM.build_B_inversion(mesh, params)
    b_inversion = nuPGCM.build_b_inversion(mesh, params, τx, τy)
end

# re-order dofs
A_inversion = A_inversion[mesh.dofs.p_inversion, mesh.dofs.p_inversion]
B_inversion = B_inversion[mesh.dofs.p_inversion, :]
b_inversion = b_inversion[mesh.dofs.p_inversion]

# preconditioner
if typeof(arch) == CPU
    P_inversion = lu(A_inversion)
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
# # set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
# # set_b!(model, x -> 0.1*exp(-((x[1] - 0.5)^2 + (x[2] + 0.5)^2 + (x[3] + α/2)^2)/2/0.1^2))
# # set_b!(model, x -> 0.1*exp(-((x[1] - 0.3)^2 + (x[2] + 0.75)^2 + (x[3] + α/2)^2)/2/0.1^2))
# # set_b!(model, x -> 0.1*exp(-((x[1] - 0.3)^2 + x[2]^2 + (x[3] + α/2)^2)/2/0.1^2))
# # set_b!(model, x -> 0.1*exp(-(x[1]^2 + (x[3] + 0.5)^2)/2/0.1^2) + 
# #                    0.1*exp(-((x[1] - 1)^2 + (x[3] + 0.5)^2)/2/0.1^2))
# # set_b!(model, x -> 1/α*x[3])
# invert!(model)
# save_state(model, "$out_dir/data/state.jld2")

# # plots
# u = model.state.u; v = model.state.v; w = model.state.w; b = model.state.b
# plot_profiles(u, v, w, b, N²; x=0.5, y=-0.75, fname="$out_dir/images/profiles_channel_basin.png")
# plot_slice(u, b, N²; bbox=[0, -α, 1, 0],  y=-0.75, cb_label=L"Zonal flow $u$",      fname="$out_dir/images/u_channel_basin.png")
# plot_slice(v, b, N²; bbox=[0, -α, 1, 0],  y=-0.75, cb_label=L"Meridional flow $v$", fname="$out_dir/images/v_channel_basin.png")
# plot_slice(w, b, N²; bbox=[0, -α, 1, 0],  y=-0.75, cb_label=L"Vertical flow $w$",   fname="$out_dir/images/w_channel_basin.png")
# plot_slice(u, b, N²; bbox=[-1, -α, 1, 0], x=0.5,   cb_label=L"Zonal flow $u$",      fname="$out_dir/images/u_channel_basin_xslice.png")
# plot_slice(v, b, N²; bbox=[-1, -α, 1, 0], x=0.5,   cb_label=L"Meridional flow $v$", fname="$out_dir/images/v_channel_basin_xslice.png")
# plot_slice(w, b, N²; bbox=[-1, -α, 1, 0], x=0.5,   cb_label=L"Vertical flow $w$",   fname="$out_dir/images/w_channel_basin_xslice.png")
# plot_slice(b, b, N²; bbox=[0, -α, 1, 0], y=-0.75, cb_label=L"Buoyancy perturbation $b'$", fname="$out_dir/images/b_channel_basin.png")
# plot_slice(u, v, b, N²; bbox=[0, -1, 1, 1], z=0.0, cb_label=L"Horizontal speed $\sqrt{u^2 + v^2}$", fname="$out_dir/images/u_sfc_channel_basin.png")
# # τx_func = interpolate_everywhere(τx, model.mesh.spaces.X_trial[1])
# # τy_func = interpolate_everywhere(τy, model.mesh.spaces.X_trial[1])
# # plot_slice(τx_func, τy_func, b, N²; bbox=[0, -1, 1, 1], z=0.0, cb_label=L"Stress $|\tau|$", fname="$out_dir/images/tau.png")

# build evolution matrices and test against saved matrices
θ = Δt/2 * α^2 * ε^2 / μϱ 
A_diff_fname = "../matrices/A_diff_$mesh_name.jld2"
A_adv_fname = "../matrices/A_adv_$mesh_name.jld2"
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
n_steps = Int(T / Δt)
# n_save = n_steps ÷ 10
# n_plot = n_steps ÷ 100
n_save = n_steps
n_plot = n_steps
run!(model; n_steps, n_save, n_plot)

println("Done.")