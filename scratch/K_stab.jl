using WriteVTK
using nuPGCM
using LinearAlgebra
using Printf
using IterativeSolvers

set_out_folder("../output")

# geometry
geom = Geometry(:circle, x->1-x[1]^2-x[2]^2, res=2)
g1 = geom.g1
g = geom.g2
H = geom.H
nσ = geom.nσ

# matrices
HM = nuPGCM.build_HM(g, H, nσ)
K_stab = nuPGCM.build_K_stab(geom)
for bdy ∈ keys(g.e)
    for i ∈ g.e[bdy]
        K_stab[i, :] .= 0
        K_stab[i, i] = 1
    end
end

# solution
u = exp.((g.p[:, 1] .+ g.p[:, 2] .+ g.p[:, 3].*H[nuPGCM.get_i_sfc.(1:g.np, nσ)])/√3)

# RHS
rhs = -(HM*u)
for bdy ∈ keys(g.e)
    for i ∈ g.e[bdy]
        rhs[i] = u[i]
    end
end

# solve
u0 = cg(K_stab, rhs, Pl=Diagonal(K_stab))

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