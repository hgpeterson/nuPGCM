using Gridap
using GridapGmsh
using Gmsh

# make simple 2D mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("mesh")
gmsh.model.geo.addPoint(0, 0, 0)
gmsh.model.geo.addPoint(1, 2, 0)  # control point
gmsh.model.geo.addPoint(2, 0, 0) 
gmsh.model.geo.addBezier([1, 2, 3])
gmsh.model.geo.addLine(3, 1)
gmsh.model.geo.addCurveLoop(1:2, 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)  # removes need for workaround
gmsh.write("mesh.msh")
gmsh.finalize()

# load mesh in Gridap
model = GmshDiscreteModel("mesh.msh")
# model = GmshDiscreteModel(joinpath(@__DIR__, "../meshes/bowl2D_1.000000e-01_5.000000e-01.msh"))
# model = GmshDiscreteModel(joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh"))
# model = GmshDiscreteModel(joinpath(@__DIR__, "../meshes/channel_basin_8.0e-02_5.0e-01.msh"))

# ✅ define lagrangian element space on mesh
reffe_l = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, reffe_l)
U = TrialFESpace(V)

# ❌ define bubble element space on mesh
reffe_b = ReferenceFE(bubble, Float64)
R = TestFESpace(model, reffe_b)
B = TrialFESpace(R)

# ✅ conformity workaround
using Gridap.ReferenceFEs
ReferenceFEs.Conformity(::GenericRefFE{Bubble}, ::Symbol) = L2Conformity()
reffe_b = ReferenceFE(bubble, Float64)
R = TestFESpace(model, reffe_b, conformity=:L2)
B = TrialFESpace(R)