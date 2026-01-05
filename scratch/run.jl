using nuPGCM
using JLD2
using Printf

include(joinpath(@__DIR__, "../meshes/channel_basin.jl"))  # for making channel_basin mesh

# ENV["JULIA_DEBUG"] = nuPGCM
ENV["JULIA_DEBUG"] = nothing

set_out_dir!(joinpath(@__DIR__, "../sims/sim015"))

# architecture
arch = GPU()

# params
ε = sqrt(1e-1)
α = 1/8
μϱ = 1
N² = 0
Δt = 1e-3
f₀ = 0.0
β = 1.0
f(x) = f₀ + β*x[2]
curved_southern_bdy = false
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
        if curved_southern_bdy
            return parabola(y, -L/2 + L_curve_channel, -L/2)
        else
            return H
        end
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
params = Parameters(ε, α, μϱ, N², Δt, f, H)
display(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)

# forcings
ν(x) = 1
κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
τ₀ = 1e-1
τˣ(x) = x[2] > -0.5 ? 0.0 : -τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
b₀ = 10  # maybe try 30 based on F18?
# b_surface(x) = 0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
# b_surface(x) = x[2] > 0 ? 0.0 : -b₀*(x[2] + 0.5)^2/0.5^2
b_surface_bc = SurfaceDirichletBC(b_surface)
# F₀ = 1
# b_flux_surface(x) = x[2] > -0.5 ? 0.0 : -F₀*sin(2π*(x[2] + 1)/0.5)
# b_surface_bc = SurfaceFluxBC(b_flux_surface)
conv_param = ConvectionParameterization(κᶜ=1e3, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=1e-2)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
# forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)  # turn off conv/eddy params
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)

# # idea:
# bottom_no_slip = DirichletBC(["bottom", "coastline", "basin bottom"], 0)
# basin_no_normal = DirichletBC(["basin", "basin top"], 0)
# top_no_normal = DirichletBC(["surface", "basin top"], 0)
# wind_stress_x = FluxBC(["surface", "basin top"], τˣ)
# wind_stress_y = FluxBC(["surface", "basin top"], τʸ)
# basin_value = DirichletBC(["surface", "basin top"], 0)
# bottom_flux = FluxBC(["surface", "basin top"], 0)
# top_flux = FluxBC(["surface", "basin top"], b_flux_surface)
# bcs = BoundaryConditions(; u=[bottom_no_slip, wind_stress_x], 
#                            v=[bottom_no_slip, basin_no_normal, wind_stress_y],
#                            w=[bottom_no_slip, top_no_normal],
#                            b=[top_flux, bottom_flux, basin_value])

function setup_model()
    # mesh
    # h = √2*α*ε
    h = √2*α*ε/2
    # h = √2*α*ε/5
    # h = √2*α*ε/10
    # mesh_name = @sprintf("channel2D_h%.2e_a%.2e", h, α)
    if curved_southern_bdy
        mesh_name = @sprintf("channel_basin_h%.2e_a%.2e", h, α)
    else
        mesh_name = @sprintf("channel_basin_h%.2e_a%.2e_vert_sb", h, α)
    end
    if !isfile(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))
        mesh_channel_basin(h, α; curved_southern_bdy)
    end
    mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

    # FE data
    # make_bc_dicts() ??
    u_diri = Dict("bottom"=>0, "coastline"=>0)
    v_diri = Dict("bottom"=>0, "coastline"=>0)
    w_diri = Dict("bottom"=>0, "coastline"=>0, "surface"=>0)
    # SurfaceDirichletBC:
    b_diri = Dict("coastline"=>b_surface, "surface"=>b_surface)
    # SurfaceFluxBC:
    # b_diri = Dict()
    spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 
    fe_data = FEData(mesh, spaces)
    @info "DOFs: $(fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw + fe_data.dofs.np)" 

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; atol=1e-6, rtol=1e-6)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    return model
end

# set up model
model = setup_model()
display(model)

# set initial buoyancy
# set_b!(model, x -> 0)  # when N² is set
# set_b!(model, x -> b₀*x[3]/α)  # when N² = 0 
# set_b!(model, x -> b₀*x[3]/α + b_surface(x)*exp(x[3]/(α/4)))  # when N² = 0 and SurfaceDirichletBC
# invert!(model)  # sync flow with buoyancy state
# save_vtk(model, ofile=@sprintf("%s/data/state_%016d.vtu", out_dir, 0))
i_step = 5200
set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", out_dir, i_step))
# set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", joinpath(@__DIR__, "channel_2D/sim008"), i_step))

# solve
T = 10*μϱ/ε^2
n_steps = Int(round(T / Δt))
n_save = 100
n_plot = Inf
# run!(model; n_steps, n_save, n_plot)
run!(model; n_steps, n_save, n_plot, i_step)

println("Done.")
