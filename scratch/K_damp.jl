using WriteVTK
using nuPGCM
using LinearAlgebra
using Printf

set_out_folder("../output")

# grid
nref = 2
g_sfc1 = Grid(Triangle(order=1), "../meshes/circle/mesh$nref.h5")
g_sfc2 = nuPGCM.add_midpoints(g_sfc1)
g1, g2, σ = nuPGCM.generate_wedge_cols(g_sfc1, g_sfc2)
nσ = length(σ)

# depth
H = FEField(x->1 - x[1]^2 - x[2]^2, g_sfc2)
Hx = DGField([∂x(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
Hy = DGField([∂y(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

# matrices
g = g2
# M = nuPGCM.mass_matrix(g)
HM = nuPGCM.build_HM(g, H, nσ)
# K_damp = nuPGCM.stiffness_matrix(g)
K_damp = nuPGCM.build_K_damp(g, H, Hx, Hy, nσ)
for bdy ∈ keys(g.e)
    for i ∈ g.e[bdy]
        K_damp[i, :] .= 0
        K_damp[i, i] = 1
    end
end
K_damp = lu(K_damp)

# solution
u = exp.((g.p[:, 1] .+ g.p[:, 2] .+ g.p[:, 3].*H[nuPGCM.get_i_sfc.(1:g.np, nσ)])/√3)
# u = exp.((g.p[:, 1] .+ g.p[:, 2] .+ g.p[:, 3])/√3)

# RHS
# rhs = -(M*u)
rhs = -(HM*u)
for bdy ∈ keys(g.e)
    for i ∈ g.e[bdy]
        rhs[i] = u[i]
    end
end

# solve
u0 = K_damp\rhs

# save
@printf("Max u  = %1.1e\n", maximum(u))
@printf("Max u₀ = %1.1e\n", maximum(u0))
@printf("Max error = %1.1e\n", maximum(abs.(u - u0)))
cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
vtk_grid("$out_folder/u", g1.p', cells) do vtk
    vtk["u"] = u[1:g1.np]
    vtk["u0"] = u0[1:g1.np]
    vtk["err"] = abs.(u[1:g1.np] - u0[1:g1.np])
end
println("$out_folder/u.vtu")