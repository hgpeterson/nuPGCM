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
N² = 0
Δt = 1e-3
params = Parameters(ε, α, μϱ, N², Δt)
show(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
# H(x) = α
# H(x) = α*(1 - x[1]^2 - x[2]^2)
H_basin(x) = α*(x[1]*(1 - x[1]))/(0.5*0.5)
H_channel(x) = -α*((x[2] + 1)*(x[2] + 0.5))/(0.25*0.25)
H(x) = x[2] > -0.75 ? max(H_channel(x), H_basin(x)) : H_channel(x)
ν(x) = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
# κ(x) = 1
# y0 = -0.5
# τˣ(x) = x[2] > y0 ? 0.0 : -(x[2] + 1)*(x[2] - y0)/(1 - y0)^2
τˣ(x) = 0
τʸ(x) = 0
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2
# b₀(x) = 0.5*(x[2]^2 - 1)
# b₀(x) = 1
T = 1e1
force_build_inversion = false
force_build_evolution = true

function setup_model()
    # mesh
    # mesh_name = "basin_flat"
    # mesh_name = "channel_basin_flat"
    mesh_name = "channel_basin"
    # h = 4e-2
    # mesh_name = @sprintf("channel_basin_%.1e_%.1e", h, α)
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

    # resolution
    p, t = nuPGCM.get_p_t(mesh.model)
    edges, _, _ = nuPGCM.all_edges(t)
    hs = [norm(p[edges[i, 1], :] - p[edges[i, 2], :]) for i ∈ axes(edges, 1)]
    hmin = minimum(hs)
    h = sum(hs) / length(hs)
    hmax = maximum(hs)
    @info @sprintf("Mesh size: %.2e (hmin = %.2e, hmax = %.2e)", h, hmin, hmax)
    dim = size(t, 2) - 1
    @info "Mesh dimension: $dim"

    # FE data
    spaces = Spaces(mesh, b₀)
    fed = FEData(mesh, spaces)
    @info "DOFs: $(fed.dofs.nu + fed.dofs.nv + fed.dofs.nw + fed.dofs.np)" 

    # build inversion matrices
    A_inversion_fname = joinpath(@__DIR__, "../matrices/A_inversion_$mesh_name.jld2")
    if force_build_inversion
        @warn "You set `force_build_inversion` to `true`, building matrices..."
        A_inversion, B_inversion, b_inversion = build_inversion_matrices(fed, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
    elseif !isfile(A_inversion_fname) 
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion, b_inversion = build_inversion_matrices(fed, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
    else
        file = jldopen(A_inversion_fname, "r")
        A_inversion = file["A_inversion"]
        close(file)
        B_inversion = nuPGCM.build_B_inversion(fed, params)
        b_inversion = nuPGCM.build_b_inversion(fed, params, τˣ, τʸ)
    end

    # re-order dofs
    A_inversion = A_inversion[fed.dofs.p_inversion, fed.dofs.p_inversion]
    B_inversion = B_inversion[fed.dofs.p_inversion, :]
    b_inversion = b_inversion[fed.dofs.p_inversion]

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

    # build evolution matrices
    A_adv, b_adv, A_diff, B_diff, b_diff = build_evolution_system(fed, params, κ; 
                                           force_build=force_build_evolution,
                                           filename=joinpath(@__DIR__, "../matrices/evolution_$mesh_name.jld2"))

    # re-order dofs
    A_adv  =  A_adv[fed.dofs.p_b, fed.dofs.p_b]
    b_adv  =  b_adv[fed.dofs.p_b]
    A_diff = A_diff[fed.dofs.p_b, fed.dofs.p_b]
    B_diff = B_diff[fed.dofs.p_b, :]
    b_diff = b_diff[fed.dofs.p_b]

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
    b_adv  = on_architecture(arch, b_adv)
    A_diff = on_architecture(arch, A_diff)
    B_diff = on_architecture(arch, B_diff)
    b_diff = on_architecture(arch, b_diff)

    # setup evolution toolkit
    evolution_toolkit = EvolutionToolkit(A_adv, P_adv, b_adv, A_diff, P_diff, B_diff, b_diff)

    # put it all together in the `model` struct
    model = rest_state_model(arch, params, fed, inversion_toolkit, evolution_toolkit)

    return model
end

# set up model
model = setup_model()

# set initial buoyancy
set_b!(model, x->b₀(x) + x[3]/α)
# set_b!(model, x->b₀(x))
# b_fe = interpolate_everywhere(b₀, model.state.b.fe_space)
# model.state.b.free_values .= b_fe.free_values
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
# n_steps = Int(round(T / Δt))
# n_save = n_steps ÷ 100
n_steps = 100
n_save = Inf
n_plot = Inf
run!(model; n_steps, n_save, n_plot)

# v_cache = plot_slice(model.state.v, model.state.b, model.params.N²; 
#                      bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Meridional flow $v$", 
#                      fname=@sprintf("%s/images/v%03d.png", out_dir, n_steps))
# vw_cache = plot_slice(model.state.v, model.state.w, model.state.b, model.params.N²; 
#                       bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Speed $\sqrt{v^2 + w^2}$", 
#                       fname=@sprintf("%s/images/vw%03d.png", out_dir, n_steps))
# b_cache = plot_slice(model.state.b, model.state.b, model.params.N²; 
#                      bbox=[-1, -model.params.α, 1, 0], x=0.5, cb_label=L"Buoyancy $b$", 
#                      fname=@sprintf("%s/images/b%03d.png", out_dir, n_steps))
# plot_slice(v_cache,  model.state.v, model.state.b; fname=@sprintf("%s/images/v%03d.png", out_dir, n_steps))
# plot_slice(vw_cache, model.state.v, model.state.w, model.state.b; fname=@sprintf("%s/images/vw%03d.png", out_dir, n_steps))
# plot_slice(b_cache,  model.state.b, model.state.b; fname=@sprintf("%s/images/b%03d.png", out_dir, n_steps))

println("Done.")