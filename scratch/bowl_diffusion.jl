using nuPGCM
using Printf

# Pure diffusion test: u = 0 throughout, N² = 0, constant isotropic κ.
# The evolution equation reduces to:
#
#   μϱ ∂ₜb = α²ε² κ ∇²b
#
# with b = 0 on the surface (Dirichlet) and no-flux everywhere else.
# Starting from a non-trivial b₀, the solution must decay monotonically
# toward b = 0. We verify that max|b| is strictly decreasing and that
# the solution reaches near-zero at long time.

set_out_dir!(joinpath(@__DIR__, "bowl_diffusion_out"))

α  = 0.5
ε  = 2e-1
μϱ = 1.0
N² = 0.0          # no background stratification → rhs_diff = 0
f(x)  = 1.0
H(x)  = α*(1 - x[1]^2 - x[2]^2)
params = Parameters(; ε, α, μϱ, N², f, H)

κ = 1.0            # constant isotropic diffusivity
forcings = Forcings(1.0, x->κ, x->κ, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))

bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
mesh      = Mesh(bowl_file)
fe_data   = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true,true,true), (false,false,true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [x -> 0.0])

# diffusion timescale: τ = μϱ / (α²ε²κ / H²) ~ μϱ H² / (α²ε² κ)
# with H ~ α, τ ~ μϱ α² / (α²ε²κ) = μϱ / (ε²κ)
τ = μϱ / (ε^2 * κ)
@info @sprintf("diffusion timescale τ = %.3e", τ)

# use a timestep that's a small fraction of τ
Δt = 1e-3 * τ
n_steps = 200
t_stop  = n_steps * Δt

ts     = BDF1(; t_start=0.0, t_stop, Δt)
evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)

# inversion toolkit is needed to build the Model type, but we will never
# call invert!, so u stays zero throughout — pure diffusion
inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
model  = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)

# initial condition: flat isopycnals
set_b!(model, x -> x[3]/α)
save_vtk_p2(model, ofile=joinpath(@__DIR__, "bowl_diffusion_out/data/b_step0000"))
@info @sprintf("step %4d  t = %.3e  max|b| = %.6e", 0, 0.0, maximum(abs, model.state.b))

# time-step the diffusion equation; u stays zero (no invert! call)
b_norms = [maximum(abs, model.state.b)]
for i in 1:n_steps
    evolve!(model, nothing, nothing)   # BDF1 with u = 0
    update_t!(ts)
    push!(b_norms, maximum(abs, model.state.b))
    if mod(i, 20) == 0
        @info @sprintf("step %4d  t = %.3e  max|b| = %.6e", i, ts.t[], b_norms[end])
        save_vtk_p2(model, ofile=joinpath(@__DIR__,
            @sprintf("bowl_diffusion_out/data/b_step%04d", i)))
    end
end

# Check 1: max|b| is monotonically non-increasing
is_monotone = all(b_norms[i] >= b_norms[i+1] for i in 1:length(b_norms)-1)
@info "max|b| monotonically decreasing: $is_monotone"

# Check 2: b has decayed significantly by the end
ratio = b_norms[end] / b_norms[1]
@info @sprintf("max|b| ratio (final/initial) = %.4e  (should be ≪ 1)", ratio)

if is_monotone && ratio < 0.01
    @info "Pure diffusion test PASSED"
else
    @warn "Pure diffusion test FAILED"
    if !is_monotone
        # find first non-monotone step
        idx = findfirst(i -> b_norms[i] < b_norms[i+1], 1:length(b_norms)-1)
        @warn @sprintf("  max|b| increased at step %d: %.6e → %.6e", idx, b_norms[idx], b_norms[idx+1])
    end
end
