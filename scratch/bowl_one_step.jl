using nuPGCM
using Printf

# One-step test: invert to get flow from b₀, then take a single BDF1 evolution
# step with advection on. Saves VTU before and after for visual inspection.

set_out_dir!(joinpath(@__DIR__, "bowl_one_step_out"))

α  = 0.5
ε  = 2e-1
μϱ = 1e1
N² = 1/α
f(x)  = 1.0 + 0.5*x[2]
H(x)  = α*(1 - x[1]^2 - x[2]^2)
params = Parameters(; ε, α, μϱ, N², f, H)

κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
forcings = Forcings(1.0, κ, κ, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))

bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
mesh      = Mesh(bowl_file)
fe_data   = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true,true,true), (false,false,true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [x -> 0.0])

Δt = 1e-4 * μϱ / (α*ε)^2
ts     = BDF1(; t_start=0.0, t_stop=Δt, Δt)
evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
model  = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)

set_b!(model, x -> x[3]/α)
invert!(model)

@info @sprintf("Before step: max|u| = %.3e  max|b| = %.3e", maximum(abs, model.state.u), maximum(abs, model.state.b))
save_vtk_p2(model, ofile=joinpath(@__DIR__, "bowl_one_step_out/data/state_before"))

evolve!(model, nothing, nothing)
update_t!(ts)

@info @sprintf("After step:  max|u| = %.3e  max|b| = %.3e", maximum(abs, model.state.u), maximum(abs, model.state.b))

nan_b = any(isnan, model.state.b)
nan_u = any(isnan, model.state.u)
@info "NaN in b: $nan_b,  NaN in u: $nan_u"

# b should still be O(1) and not blow up in a single step
b_ok = maximum(abs, model.state.b) < 10.0
@info "b magnitude sane: $b_ok"

save_vtk_p2(model, ofile=joinpath(@__DIR__, "bowl_one_step_out/data/state_after"))

if !nan_b && !nan_u && b_ok
    @info "Single-step advection test PASSED"
else
    @warn "Single-step advection test FAILED"
end
