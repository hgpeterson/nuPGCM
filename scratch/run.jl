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
function H(xyz)
    x = xyz[1]
    y = xyz[2]

    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = L_channel/4 # length of flat part of channel
    L_curve_channel = (L_channel - L_flat_channel)/2 # length of each curved part of channel
    W_flat_basin = W/2 # width of flat part of basin
    W_curve_basin = (W - W_flat_basin)/2 # width of each curved part of basin
    L_curve_basin = W_curve_basin # length of curved end of basin
    H = α*W

    # parabola that has a maximum of H at x_max and a 0 at x_zero
    parabola(x, x_max, x_zero) = H*(1 - ((x - x_max)/(x_zero - x_max))^2)

    function H_basin(x)
        if 0 ≤ x ≤ W_curve_basin
            return parabola(x, W_curve_basin, 0)
        elseif x ≤ W_curve_basin + W_flat_basin
            return H
        elseif x ≤ W
            return parabola(x, W_curve_basin + W_flat_basin, W)
        else
            throw(ArgumentError("x out of bounds"))
        end
    end

    if -L/2 ≤ y ≤ -L/2 + L_curve_channel
        return parabola(y, -L/2 + L_curve_channel, -L/2)
    elseif y ≤ -L/2 + L_curve_channel + L_flat_channel
        return H
    elseif y ≤ -L/2 + L_channel
        H_channel = parabola(y, -L/2 + L_curve_channel + L_flat_channel, -L/2 + L_channel)
        return max(H_channel, H_basin(x))
    elseif y ≤ L/2 - L_curve_basin
        return H_basin(x)
    elseif y ≤ L/2
        if 0 ≤ x ≤ W_curve_basin
            x₀ = W_curve_basin
            y₀ = L/2 - L_curve_basin
            r = √( (x - x₀)^2 + (y - y₀)^2 )
            return parabola(r, 0, W_curve_basin)
        elseif W_curve_basin ≤ x ≤ W_curve_basin + W_flat_basin
            return parabola(y, L/2 - L_curve_basin, L/2)
        elseif x ≤ W
            x₀ = W_curve_basin + W_flat_basin
            y₀ = L/2 - L_curve_basin
            r = √( (x - x₀)^2 + (y - y₀)^2 )
            return parabola(r, 0, W_curve_basin)
        else
            throw(ArgumentError("x out of bounds"))
        end
    else
        throw(ArgumentError("y out of bounds"))
    end
end
ν = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
# κ(x) = 1
τˣ(x) = x[2] > -0.5 ? 0.0 : -1e-4*(x[2] + 1)*(x[2] + 0.5)/(1 + 0.5)^2
τʸ(x) = 0
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2
force_build_inversion = true
force_build_evolution = true

function setup_model()
    # mesh
    h = 8e-2
    mesh_name = @sprintf("channel_basin_h%.2e_a%.2e", h, α)
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

    # build evolution matrices
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

    return model
end

# set up model
model = setup_model()

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