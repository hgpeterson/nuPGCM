using nuPGCM
using PyPlot
import Gmsh: gmsh

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

gmsh.initialize()

gmsh.model.add("t17")

# create a square
gmsh.model.occ.addRectangle(-2, -2, 0, 4, 4)
gmsh.model.occ.synchronize()

# merge a post-processing view containing the target anisotropic mesh sizes
gmsh.merge("t17_bgmesh.pos")

# apply the view as the current background mesh
bg_field = gmsh.model.mesh.field.add("PostView")
gmsh.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

# use bamg
gmsh.option.setNumber("Mesh.SmoothRatio", 3)
gmsh.option.setNumber("Mesh.AnisoMax", 1000)
gmsh.option.setNumber("Mesh.Algorithm", 7)

gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")

gmsh.finalize()

# load
p, t, e = load_gmesh()

# plot
tplot(p, t)
axis("equal")
savefig("mesh.png")