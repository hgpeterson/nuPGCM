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
arch = GPU()

# params
ε = 1e-1
α = 1/2
μϱ = 1
N² = 0
Δt = 1e-3
κᶜ = 100
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
params = Parameters(ε, α, μϱ, N², Δt, κᶜ, f, H)
display(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)

# forcings
ν(x) = 1
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
τ₀ = 1e-1
τˣ(x) = x[2] > -0.5 ? 0.0 : -τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b₀)

function setup_model()
    # mesh
    h = 8e-2
    mesh_name = @sprintf("channel_basin_h%.2e_a%.2e", h, α)
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

    # FE data
    spaces = Spaces(mesh, b₀)
    fe_data = FEData(mesh, spaces, forcings)
    @info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; atol=1e-6, rtol=1e-6)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = rest_state_model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    return model
end

# # set up model
# model = setup_model()

# set initial buoyancy
set_b!(model, x->b₀(x) + x[3]/α)
nuPGCM.update_κᵥ!(model, model.state.b)  # reset κᵥ
invert!(model) # sync flow with buoyancy state
save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))

# solve
T = 10*μϱ/ε^2
n_steps = Int(round(T / Δt))
# n_save = n_steps ÷ 100
n_save = 100
n_plot = Inf
run!(model; n_steps, n_save, n_plot)

println("Done.")