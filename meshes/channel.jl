using Gmsh: gmsh

gmsh.initialize()

gmsh.model.add("channel")

h = 0.08
gmsh.model.occ.addPoint(0, -0.25, 0)
gmsh.model.occ.addPoint(0, 0, -2*1/2) # control point
gmsh.model.occ.addPoint(0, 0.25, 0)
gmsh.model.occ.addBezier([1, 2, 3], 1) # parabola
gmsh.model.occ.addLine(3, 1, 2)
gmsh.model.occ.addCurveLoop(1:2, 1)
gmsh.model.occ.addPlaneSurface([1], 1)
gmsh.model.occ.extrude([(2, 1)], 1, 0, 0)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

# periodic boundary condition
translation = [1, 0, 0, 1, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(2, [4], [1], translation)
gmsh.model.occ.synchronize()

# define bottom, surface, and interior
gmsh.model.addPhysicalGroup(0, [1, 3, 4, 5], 1, "bot")
gmsh.model.addPhysicalGroup(1, [1, 3, 4, 5], 1, "bot")
gmsh.model.addPhysicalGroup(1, [2, 6], 2, "sfc")
gmsh.model.addPhysicalGroup(2, [2], 1, "bot")
gmsh.model.addPhysicalGroup(2, [3], 2, "sfc")
gmsh.model.addPhysicalGroup(2, [1, 4], 3, "int")
gmsh.model.addPhysicalGroup(3, [1], 3, "int")

# set resolution
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

gmsh.model.mesh.generate(3)
gmsh.write("channel.msh")
gmsh.finalize()