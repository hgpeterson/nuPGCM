using Gmsh: gmsh

# init
gmsh.initialize()
gmsh.model.add("channelbasin")

# params
h = 0.1
α = 0.5
l = 0.2
R = 1

# points
gmsh.model.occ.addPoint(R,     0,  l, 1)
gmsh.model.occ.addPoint(R - α, 0,  0, 2)
gmsh.model.occ.addPoint(R,     0, -l, 3)

# revolve bottom curve around z-axis (makes surface 1)
gmsh.model.occ.addBezier([1, 2, 3], 1)
s1 = gmsh.model.occ.revolve([(1, 1)], 0, 0, 0, 0, 0, 1, 2π)
println(s1)

# revolve top line around z-axis (makes surface 2)
gmsh.model.occ.addLine(1, 3, 4)
s2 = gmsh.model.occ.revolve([(1, 4)], 0, 0, 0, 0, 0, 1, 2π)
println(s2)

# l1 = gmsh.model.occ.addCurveLoop([1, 5])
# gmsh.model.occ.addSurfaceFilling(l1, 3)
# l2 = gmsh.model.occ.addCurveLoop([4, 8])
# gmsh.model.occ.addSurfaceFilling(l2, 4)

# # define volume
# gmsh.model.occ.addSurfaceLoop(1:2, 1)
# gmsh.model.occ.addVolume([1], 1)

# mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
gmsh.model.mesh.generate(2)
# gmsh.model.mesh.generate(3)
gmsh.write("channelbasin.msh")
gmsh.finalize()