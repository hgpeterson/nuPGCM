using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot

pygui(false)
plt.style.use("../nuPGCM/plots.mplstyle")
plt.close("all")

# model
model = GmshDiscreteModel("bowl.msh")
# writevtk(model, "model")
# error()

# reference FE 
reffe_ux = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, 1; space=:P)
reffe_p  = ReferenceFE(lagrangian, Float64, 0; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot", "corners"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Q  = TestFESpace(model, reffe_p,  conformity=:L2, constraint=:zeromean)
# Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
Y = MultiFieldFESpace([Vx, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0, 0])
Uz = TrialFESpace(Vz, [0, 0, 0])
P  = TrialFESpace(Q)
X  = MultiFieldFESpace([Ux, Uz, P])

# triangulation and integration measure
degree = 2
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)
Γ = BoundaryTriangulation(model, tags=["top"])
dΓ = Measure(Γ, degree)

# gradients 
x = VectorValue(1.0, 0.0)
z = VectorValue(0.0, 1.0)
∂x(u) = x⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
δ = 0.1
H(x) = sqrt(2 - x^2) - 1
b(x) = δ*exp(-(x[2] + H(x[1]))/δ)

# bilinear and linear form
a((ux, uz, p), (vx, vz, q)) = ∫( ∂z(vx)*∂z(ux) - ∂x(vx)*p - ∂z(vz)*p + q*∂x(ux) + q*∂z(uz) )dΩ
l((vx, vz, q)) = ∫( b*vz )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ux, uz, p = solve(op)

# export to vtk
writevtk(Ωₕ, "results", cellfields=["ux"=>ux, "uz"=>uz, "p"=>p])

# n = 30
# x = range(0,  1,  n)
# z = range(0,  1,  n)

# uxdata = zeros(n,  n)
# uzdata = zeros(n,  n)
# pdata = zeros(n,  n)
# for i=1:n
#     for j=1:n
#         uxdata[i,  j] = evaluate(ux,  Point(x[i],  z[j]))
#         uzdata[i,  j] = evaluate(uz,  Point(x[i],  z[j]))
#         pdata[i,  j] = evaluate(p,  Point(x[i],  z[j]))
#     end
# end

# function quickplot(x,  z,  u,  clabel,  fname)
#     fig,  ax = subplots(1)
#     vmax = maximum(u)
#     im = ax.pcolormesh(x,  z,  u',  shading="auto",  cmap="RdBu_r",  vmin=-vmax,  vmax=vmax)
#     colorbar(im,  ax=ax,  label=clabel)
#     ax.axis("equal")
#     ax.set_xlabel(L"x")
#     ax.set_ylabel(L"z")
#     savefig(fname)
#     println(fname)
#     plt.close()
# end

# quickplot(x,  z,  uxdata,  L"u_1",  "u1.png")
# quickplot(x,  z,  uzdata,  L"u_2",  "u2.png")
# quickplot(x,  z,  pdata,  L"p",  "p.png")
