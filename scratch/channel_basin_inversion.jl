using nuPGCM
using Gridap
using JLD2
using Printf
using PyPlot

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

# for making mesh
include(joinpath(@__DIR__, "../meshes/channel_basin.jl"))  

ENV["JULIA_DEBUG"] = nuPGCM
ENABLE_TIMING[] = true

i_sim = 44
set_out_dir!(joinpath(@__DIR__, @sprintf("../sims/sim%03dc", i_sim)))

# architecture
arch = GPU()

# params

Ω = 2π/86400  # s⁻¹
a = 6.371e6  # m
β = 2Ω/a  # m⁻¹ s⁻¹
L = 2π*a*60/360  # m
f₀ = β*L  # s⁻¹
H₀ = 4e3  # m
κ₀ = 1e-5  # m² s⁻¹
Kₑ = 1000  # m² s⁻¹
N₀ = 1e-3  # s⁻¹
ρ₀ = 1035  # kg m⁻³
α_T = 2e-4  # °C⁻¹
g = 9.81  # m s⁻²
ν₀ = Kₑ*f₀^2/N₀^2  # m² s⁻¹
τ₀ = ρ₀*N₀^2*H₀^3/L  # N m⁻²
b₀ = g*α_T*30/(N₀^2*H₀)

κ₀ *= 10

ε = sqrt(ν₀/f₀/H₀^2)
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2

t₀ = 1/f₀/ϱ  # s
@info "scales" b₀ ν₀ τ₀ t₀

μϱ = μ*ϱ
α = 1/4
N² = 0
Δt = 4*86400/t₀
f(x) = x[2]
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
        if 0 ≤ x ≤ W
            return parabola(x, W/2, 0)
        else
            throw(ArgumentError("x out of bounds"))
        end
    end

    if -L/2 ≤ y ≤ -L/2 + L_curve_channel + L_flat_channel
        return H
    elseif y ≤ -L/2 + L_channel
        H_channel = parabola(y, -L/2 + L_curve_channel + L_flat_channel, -L/2 + L_channel)
        return max(H_channel, H_basin(x))
    elseif y ≤ L/2
        return H_basin(x)
    else
        throw(ArgumentError("y out of bounds"))
    end
end
params = Parameters(ε, α, μϱ, N², Δt, f, H)
display(params)
@info @sprintf("Diffusion timescale: %.2e", μϱ/ε^2)

# forcings
ν(x) = 1
κ_B = 1e2
κ_I = 1
# d = 500/4000 * α
d = 500/2000 * α  # 2x larger decay scale
κₕ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
κᵥ(x) = κ_I + (κ_B - κ_I)*exp(-(x[3] + H(x))/d)
τˣ(x) = x[2] > -0.5 ? 0.0 : -0.2/τ₀*(x[2] + 1)*(x[2] + 0.5)/0.25^2
τʸ(x) = 0
b_surface(x) = x[2] > 0 ? 0.0 : -b₀*x[2]^2
b_surface_bc = SurfaceDirichletBC(b_surface)
conv_param = ConvectionParameterization(κᶜ=0.2/κ₀, N²min=1e-3)
eddy_param = EddyParameterization(f=f, N²min=1e-3)
forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc; conv_param, eddy_param)
display(forcings)
display(forcings.conv_param)
display(forcings.eddy_param)

# mesh
h = 2e-2
mesh_name = @sprintf("channel_basin_no_flat_h%.2e_a%.2e", h, α)
if !isfile(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))
    mesh_channel_basin_no_flat(h, α)
end
mesh = Mesh(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

# FE data
u_diri_tags = ["bottom", "coastline", "surface"]
u_diri_vals = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
u_diri_masks = [(true, true, true), (true, true, true), (false, false, true)]
b_diri_tags = ["coastline", "surface"]
b_diri_vals = [b_surface, b_surface]
spaces = Spaces(mesh; u_diri_tags, u_diri_vals, u_diri_masks, b_diri_tags, b_diri_vals, b_order=1) 
fe_data = FEData(mesh, spaces)
display(fe_data.dofs)

# setup inversion toolkit
itmax = 8000
inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; itmax)

# put it all together in the `model` struct
model = Model(arch, params, forcings, fe_data, inversion_toolkit)

# set buoyancy from file
i_step = 2500
set_state_from_file!(model.state, @sprintf("%s/data/state_%016d.jld2", out_dir, i_step))

# invert
invert!(model)

# plot residuals
res = model.inversion.solver.solver.stats.residuals
fig, ax = plt.subplots(1)
ax.loglog(res)
ax.set_xlabel("Iterations")
ax.set_ylabel("Residual")
savefig(joinpath(@__DIR__, "residuals.png"))
println(joinpath(@__DIR__, "residuals.png"))
plt.close()

# jldopen(joinpath(@__DIR__, "data/state_n58378.jld2"), "r") do file
#     u_data = file["u"]
#     U_trial, P_trial = model.fe_data.spaces.X_trial
#     B_trial = model.fe_data.spaces.B_trial
#     u0 = FEFunction(U_trial, u_data)
#     u = model.state.u
#     dΩ = model.fe_data.mesh.dΩ
#     println("its\tL2\tL∞")
#     @printf("%d\t%.3e\t%.3e\n", itmax, √(sum(∫( (u - u0)⋅(u - u0) )dΩ)), maximum(abs.(u.free_values - u0.free_values)))
# end

# # plot errors
# data = [ 1000   1.250e-01  4.526e+00  # its, L2, L∞
#          2000   7.859e-02  4.342e+00
#          4000   4.412e-02  3.497e+00
#          8000   2.105e-02  2.180e+00
#          16000  8.402e-03  9.026e-01
#          32000  1.417e-03  1.524e-01]
# fig, ax = plt.subplots(1)
# ax.loglog(data[:, 1], data[:, 2], "o-", label=L"L_2")
# ax.loglog(data[:, 1], data[:, 3], "o-", label=L"L_\infty")
# ax.legend()
# ax.set_xlabel("Iterations")
# ax.set_ylabel("Error")
# savefig(joinpath(@__DIR__, "errors.png"))
# println(joinpath(@__DIR__, "errors.png"))
# plt.close()