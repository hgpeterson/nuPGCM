using Gmsh: gmsh

gmsh.initialize()

gmsh.model.add("channel_basin_cart")

# params
h = 0.04
L = 2
W = 1
L_channel = L/4
y_channel = -L/2 + L_channel/2 # channel center
α = 1/2 # H/W

# channel curve
gmsh.model.occ.addPoint(0, -L/2, 0)
gmsh.model.occ.addPoint(0, y_channel, -2α*W) # control point
gmsh.model.occ.addPoint(0, -L/2 + L_channel, 0)
gmsh.model.occ.addBezier([1, 2, 3], 1) # parabola

# turn it into a closed surface
gmsh.model.occ.addLine(3, 1, 2)
gmsh.model.occ.addCurveLoop(1:2, 1)
gmsh.model.occ.addPlaneSurface([1], 1)

# extrude to x = W
gmsh.model.occ.extrude([(2, 1)], W, 0, 0)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.04)

# periodic boundary condition
translation = [1, 0, 0, W, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(2, [4], [1], translation)
gmsh.model.occ.synchronize()

# basin curve
p1 = gmsh.model.occ.addPoint(0, y_channel, 0)
p2 = gmsh.model.occ.addPoint(W/2, y_channel, -2α*W) # control point
p3 = gmsh.model.occ.addPoint(W, y_channel, 0)
l1 = gmsh.model.occ.addBezier([p1, p2, p3]) # parabola

# turn it into a closed surface
l2 = gmsh.model.occ.addLine(p3, p1)
c1 = gmsh.model.occ.addCurveLoop([l1, l2])
s1 = gmsh.model.occ.addPlaneSurface([c1])

# extrude to y = L/2
gmsh.model.occ.extrude([(2, s1)], 0, L/2 - y_channel, 0)
gmsh.model.occ.synchronize()

# fuse the two volumes
gmsh.model.occ.fuse([(3, 1)], [(3, 2)])
gmsh.model.occ.synchronize()

# define bottom, surface, and interior
gmsh.model.addPhysicalGroup(0, 8:13, 1, "bot")
gmsh.model.addPhysicalGroup(1, 1:10, 1, "bot")
gmsh.model.addPhysicalGroup(2, [1, 3, 6], 1, "bot")
gmsh.model.addPhysicalGroup(2, [5], 2, "sfc")
gmsh.model.addPhysicalGroup(2, [2, 4], 3, "int")
gmsh.model.addPhysicalGroup(3, [1], 3, "int")

# set resolution
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

gmsh.model.mesh.generate(3)
gmsh.write("channel_basin_cart.msh")
gmsh.write("channel_basin_cart.vtk")
gmsh.finalize()