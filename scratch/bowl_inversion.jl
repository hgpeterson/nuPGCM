using nuPGCM
using Printf

set_out_dir!(joinpath(@__DIR__, "bowl_inversion_out"))

arch = CPU()

ε  = 2e-1
α  = 0.5
μϱ = 1e1
N² = 1/α
f(x)  = 1.0 + 0.5*x[2]
H(x)  = α*(1 - x[1]^2 - x[2]^2)
params = Parameters(; ε, α, μϱ, N², f, H)

ν = 1.0
forcings = Forcings(ν, x->1e-2, x->1e-2, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))

bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
mesh = Mesh(bowl_file)

fe_data = FEData(mesh;
    u_diri_tags  = ["bottom", "surface"],
    u_diri_masks = [(true,true,true), (false,false,true)],
    b_diri_tags  = ["surface"],
    b_diri_vals  = [x -> 0.0])

@info "DOFs" fe_data.nu fe_data.np fe_data.nb

inv_tk = InversionToolkit(arch, fe_data, params, forcings)
model  = Model(arch, params, forcings, fe_data, inv_tk)

# bottom-enhanced buoyancy perturbation
set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
@info "b stats" minimum(model.state.b) maximum(model.state.b)

invert!(model)
@info "u stats" minimum(model.state.u) maximum(model.state.u)
@info "w stats" minimum(model.state.u[3:3:end]) maximum(model.state.u[3:3:end])

save_vtk_p2(model, ofile=joinpath(@__DIR__, "bowl_inversion_out/data/bowl_inversion"))
@info "Done. Open scratch/bowl_inversion_out/data/bowl_inversion.vtu in ParaView."
