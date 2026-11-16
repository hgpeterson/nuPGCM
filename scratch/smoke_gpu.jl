# GPU smoke test. The test suite is CPU-only, so this exercises the CUDA
# extension end to end (upload, ILU(0), block-triangular FGMRES inversion, a few
# timesteps) and compares against the same run on the CPU.
using nuPGCM, CUDA, Printf, LinearAlgebra

CUDA.versioninfo()

PROJ_PATH = "/resnick/groups/oceanphysics/henry/nuPGCM-ferrite"
set_out_dir!("/home/hppeters/scratch/smoke_gpu")

mesh = Mesh(joinpath(PROJ_PATH, "meshes/channel_basin_flat_h1.00e-01_a5.00e-01.msh"))
params = Parameters(; ε=0.2, α=0.5, μϱ=10.0, N²=2.0, f=x->1.0, H=x->1.0)
forcings = Forcings(1.0, x->1e-2, x->1e-2, x->0.05, x->0.0, SurfaceDirichletBC(x->0.0))
fe_data = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true, true, true), (false, false, true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [x -> 0.0],
    b_order = 1)

function go(arch)
    @info "=== $arch ==="
    Δt = 1e-3
    ts = BDF1(; t_start=0.0, t_stop=5Δt, Δt)
    inv_tk = InversionToolkit(arch, fe_data, params, forcings; itmax=800, atol=1e-10, rtol=1e-10)
    evo_tk = EvolutionToolkit(arch, fe_data, params, forcings, ts)
    model = Model(arch, params, forcings, fe_data, inv_tk, evo_tk, ts)
    set_b!(model, x -> 0.1*x[3])
    invert!(model)
    run!(model; n_info=1)
    return Array(model.state.u), Array(model.state.b)
end

u_c, b_c = go(CPU())
u_g, b_g = go(GPU())
rel(a, b) = norm(a - b)/norm(b)
@info @sprintf("GPU vs CPU: u %.2e  b %.2e", rel(u_g, u_c), rel(b_g, b_c))
# CPU evolution is LU, GPU is CG at rtol 1e-6
@assert rel(u_g, u_c) < 1e-5 && rel(b_g, b_c) < 1e-3
@info "GPU smoke test OK"
