using nuPGCM
using Gridap
using JLD2
using Printf
using PyPlot

pygui(false)
plt.close("all")
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))

include(joinpath(@__DIR__, "../meshes/mesh_bowl2D.jl"))
set_out_dir!(@__DIR__)

# # params/funcs
# arch = CPU()
# ε = 0.1
# α = 1/2
# μϱ = 1
# N² = 1/α
# Δt = 1e-2/2^0
# f(x) = 1
# H(x) = α*(1 - x[1]^2 - x[2]^2)
# params = Parameters(ε, α, μϱ, N², Δt, f, H)
# ν = 1
# κₕ(x) = 1 + (100 - 1)*exp(-(x[3] + H(x))/(0.2*α))
# κᵥ(x) = 1 + (100 - 1)*exp(-(x[3] + H(x))/(0.2*α))
# τˣ(x) = 0
# τʸ(x) = 0
# b_surface(x) = 0
# b_surface_bc = SurfaceDirichletBC(b_surface)
# forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
# n_steps = Int64(0.04 ÷ Δt)

# # h = 0.01
# # meshfile = joinpath(@__DIR__, @sprintf("../meshes/bowl2D_%e_%e.msh", h, α))
# # if !isfile(meshfile)
# #     generate_bowl_mesh_2D(h, α)
# # end
# # mesh = Mesh(meshfile)

# # # FE data
# # u_diri_tags = ["bottom", "coastline", "surface"]
# # u_diri_vals = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
# # u_diri_masks = [(true, true, true), (true, true, true), (false, false, true)]
# # b_diri_tags = ["coastline", "surface"]
# # b_diri_vals = [b_surface, b_surface]
# # spaces = Spaces(mesh; u_diri_tags, u_diri_vals, u_diri_masks, b_diri_tags, b_diri_vals, b_order=1) 
# # fe_data = FEData(mesh, spaces)

# # # setup inversion toolkit
# # inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings)

# # build evolution system
# # order = 1
# order = 2
# evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings; order) 

# # put it all together in the `model` struct
# model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

# # solve
# run!(model; n_steps)

# # # plot
# # save_vtk(model)

# # save
# save_state(model, @sprintf("%s/data/state_o%d_dt%.3e.jld2", out_dir, order, Δt))

Δts = collect(1e-2 * (0.5 .^ (0:5)))

function calculate_GTE(order)
    B_trial = model.fe_data.spaces.B_trial
    dΩ = model.fe_data.mesh.dΩ

    # "true" solution as the one with smallest timestep
    d = jldopen(@sprintf("%s/data/state_o%d_dt%.3e.jld2", out_dir, order, Δts[end]), "r")
    b0 = FEFunction(B_trial, d["b"])
    close(d)

    # compute errors 
    errors = zeros(length(Δts)-1)
    for i in eachindex(errors)
        d = jldopen(@sprintf("%s/data/state_o%d_dt%.3e.jld2", out_dir, order, Δts[i]), "r")
        b = FEFunction(B_trial, d["b"])
        close(d)
        errors[i] = sum(∫( (b - b0)*(b - b0) )dΩ)
    end

    return errors
end

function plot_GTE()
    errors1 = calculate_GTE(1)
    errors2 = calculate_GTE(2)

    # plot
    fig, ax = plt.subplots(1)
    xmin = Δts[end-1]
    xmax = Δts[1]
    Δx = xmax - xmin
    ymin = min(errors1[end], errors2[end])
    ymax = max(errors1[1], errors2[end])
    # ax.loglog([xmin, xmax], [1.2*ymin, 1.2*ymin*Δx^-1], "k-",  label=L"$\Delta t$")
    # ax.loglog([xmin, xmax], [1.2*ymin, 1.2*ymin*Δx^-2], "k--", label=L"$\Delta t^2$")
    ax.loglog([xmin, xmax], [1e-08, 1e-08/Δx^1], "k-",  label=L"$\Delta t$")
    ax.loglog([xmin, xmax], [1e-11, 1e-11/Δx^2], "k--", label=L"$\Delta t^2$")
    ax.loglog(Δts[1:end-1], errors1, "o", label="BDF1")
    ax.loglog(Δts[1:end-1], errors2, "o", label="BDF2")
    ax.set_xlabel(L"\Delta t")
    ax.set_ylabel(L"\Vert b^n_{\Delta t} - b^n_{3.125 \times 10^{-4}} \Vert_{L^2}")
    ax.set_title("Global truncation error")
    # ax.set_xlim(0.8*xmin, 1.2*xmax)
    # ax.set_ylim(0.8*ymin, 1.2*ymax)
    ax.set_xlim(5e-4, 2e-2)
    ax.set_ylim(1e-13, 1e-5)
    ax.legend()
    savefig("$out_dir/images/gte.png")
    println("$out_dir/images/gte.png")
    plt.close()

    return errors1, errors2
end

plot_GTE()
