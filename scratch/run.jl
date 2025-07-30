using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

set_out_dir!(joinpath(@__DIR__, ""))

# architecture
arch = CPU()

# params/funcs
ε = 1e-1
α = 1/2
μϱ = 1
N² = 1/α
Δt = 1e-3
params = Parameters(ε, α, μϱ, N², Δt)
show(params)
@info @sprintf("Diffusion timescale: %.2e", α / (α^2 * ε^2 / μϱ))
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
# H(x) = α*(1 - x[1]^2 - x[2]^2)
H_basin(x) = α*(x[1]*(1 - x[1]))/(0.5*0.5)
H_channel(x) = -α*((x[2] + 1)*(x[2] + 0.5))/(0.25*0.25)
H(x) = x[2] > -0.75 ? max(H_channel(x), H_basin(x)) : H_channel(x)
ν(x) = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
# κ(x) = 1
y0 = -0.5
τˣ(x) = x[2] > y0 ? 0.0 : -(x[2] + 1)*(x[2] - y0)/(1 - y0)^2
τʸ(x) = 0
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2
T = 1e1
force_build_inversion_matrices = false
force_build_evolution_matrices = false

function setup_model()
    # mesh
    # mesh_name = "channel_basin_cart_h0.01_a0.5"
    # mesh_name = "channel_basin_cart_h0.08_a0.5"
    mesh_name = "channel_basin_cart_more_sfc"
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"), b₀)
    @info "DOFs: $(mesh.dofs.nu + mesh.dofs.nv + mesh.dofs.nw + mesh.dofs.np)" 
    p, t = nuPGCM.get_p_t(mesh.model)
    edges, _, _ = nuPGCM.all_edges(t)
    hs = [norm(p[edges[i, 1], :] - p[edges[i, 2], :]) for i ∈ axes(edges, 1)]
    hmin = minimum(hs)
    h = sum(hs) / length(hs)
    hmax = maximum(hs)
    @info @sprintf("Mesh size: %.2f (hmin = %.2f, hmax = %.2f)", h, hmin, hmax)
    dim = size(t, 2) - 1
    @info "Mesh dimension: $dim"

    # build inversion matrices
    A_inversion_fname = joinpath(@__DIR__, "../matrices/A_inversion_$mesh_name.jld2")
    if force_build_inversion_matrices
        @warn "You set `force_build_inversion_matrices` to `true`, building matrices..."
        A_inversion, B_inversion, b_inversion = build_inversion_matrices(mesh, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
    elseif !isfile(A_inversion_fname) 
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion, b_inversion = build_inversion_matrices(mesh, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
    else
        file = jldopen(A_inversion_fname, "r")
        A_inversion = file["A_inversion"]
        close(file)
        B_inversion = nuPGCM.build_B_inversion(mesh, params)
        b_inversion = nuPGCM.build_b_inversion(mesh, params, τˣ, τʸ)
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

    # build evolution matrices and test against saved matrices
    θ = Δt/2 * α^2 * ε^2 / μϱ 
    A_diff_fname = joinpath(@__DIR__, "../matrices/A_diff_$mesh_name.jld2")
    A_adv_fname = joinpath(@__DIR__, "../matrices/A_adv_$mesh_name.jld2")
    if force_build_evolution_matrices
        @warn "You set `force_build_evolution_matrices` to `true`, building matrices..."
        A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ; 
                                            A_adv_ofile=A_adv_fname, A_diff_ofile=A_diff_fname)
    elseif !isfile(A_diff_fname) || !isfile(A_adv_fname) 
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

    return model
end

model = setup_model()

# set_b!(model, x->0)
# κ_fe = interpolate_everywhere(κ, model.mesh.spaces.X_trial[1])
# b = interpolate_everywhere(x->0, model.mesh.spaces.B_trial)
# plot_slice(κ_fe, b, 0; 
# # plot_slice(κ, b, 0; 
#            bbox=[-1, -α, 1, 0], x=0.5, cb_label=L"Diffusivity $\kappa$", 
#            fname=@sprintf("%s/images/kappa.png", out_dir))

# set initial buoyancy
set_b!(model, b₀)
# set_b!(model, x->b₀(x)*(α + x[3]))
# set_b!(model, x->0)
# save_vtk(model; ofile=joinpath(@__DIR__, "data/state_0.vtu"))
plot_slice(model.state.b, model.state.b, model.params.N²; 
           bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Buoyancy $b$", 
           fname=@sprintf("%s/images/b%03d.png", out_dir, 0))

# solve
n_steps = Int(round(T / Δt))
n_save = n_steps ÷ 100
n_plot = Inf
run!(model; n_steps, n_save, n_plot)

v_cache = plot_slice(model.state.v, model.state.b, model.params.N²; 
                     bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Meridional flow $v$", 
                     fname=@sprintf("%s/images/v%03d.png", out_dir, n_steps))
vw_cache = plot_slice(model.state.v, model.state.w, model.state.b, model.params.N²; 
                      bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Speed $\sqrt{v^2 + w^2}$", 
                      fname=@sprintf("%s/images/vw%03d.png", out_dir, n_steps))
b_cache = plot_slice(model.state.b, model.state.b, model.params.N²; 
                     bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Buoyancy $b$", 
                     fname=@sprintf("%s/images/b%03d.png", out_dir, n_steps))
# plot_slice(v_cache,  model.state.v, model.state.b; fname=@sprintf("%s/images/v%03d.png", out_dir, n_steps))
# plot_slice(vw_cache, model.state.v, model.state.w, model.state.b; fname=@sprintf("%s/images/vw%03d.png", out_dir, n_steps))
# plot_slice(b_cache,  model.state.b, model.state.b; fname=@sprintf("%s/images/b%03d.png", out_dir, n_steps))

println("Done.")