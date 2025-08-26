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
T = μϱ/ε^2
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
H_basin(x) = α*(x[1]*(1 - x[1]))/(0.5*0.5)
H_channel(x) = -α*((x[2] + 1)*(x[2] + 0.5))/(0.25*0.25)
H(x) = x[2] > -0.75 ? max(H_channel(x), H_basin(x)) : H_channel(x)
ν(x) = 1
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
τˣ(x) = x[2] > -0.5 ? 0.0 : -1e-4*(x[2] + 1)*(x[2] + 0.5)/(1 + 0.5)^2
τʸ(x) = 0
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2
force_build_inversion = false
force_build_evolution = true

function setup_model()
    # mesh
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
    # # set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
    # # set_b!(model, x -> 0.1*exp(-((x[1] - 0.5)^2 + (x[2] + 0.5)^2 + (x[3] + α/2)^2)/2/0.1^2))
    # # set_b!(model, x -> 0.1*exp(-((x[1] - 0.3)^2 + (x[2] + 0.75)^2 + (x[3] + α/2)^2)/2/0.1^2))
    # # set_b!(model, x -> 0.1*exp(-((x[1] - 0.3)^2 + x[2]^2 + (x[3] + α/2)^2)/2/0.1^2))
    # # set_b!(model, x -> 0.1*exp(-(x[1]^2 + (x[3] + 0.5)^2)/2/0.1^2) + 
    # #                    0.1*exp(-((x[1] - 1)^2 + (x[3] + 0.5)^2)/2/0.1^2))
    # # set_b!(model, x -> 1/α*x[3])
    # invert!(model)
    # save_state(model, "$out_dir/data/state.jld2")

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, κₕ, κᵥ; 
                            force_build=force_build_evolution,
                            filename=joinpath(@__DIR__, "../matrices/evolution_$mesh_name.jld2"))

    # put it all together in the `model` struct
    model = rest_state_model(arch, params, fe_data, inversion_toolkit, evolution_toolkit)

    return model
end

# # set up model
# model = setup_model()

# set initial buoyancy
set_b!(model, x->b₀(x) + x[3]/α)
invert!(model) # sync flow with buoyancy state
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
n_steps = Int(round(T / Δt))
# n_steps = 100
# n_save = n_steps ÷ 100
n_save = 100
n_plot = Inf
run!(model; n_steps, n_save, n_plot)

println("Done.")