using Gmsh: gmsh

gmsh.initialize()

gmsh.model.add("periodic_square")

gmsh.model.occ.addPoint(0, 0, -1)
gmsh.model.occ.addPoint(1, 0, -1)
gmsh.model.occ.addPoint(1, 0, 0)
gmsh.model.occ.addPoint(0, 0, 0)
gmsh.model.occ.addLine(1, 2)
gmsh.model.occ.addLine(2, 3)
gmsh.model.occ.addLine(3, 4)
gmsh.model.occ.addLine(4, 1)
gmsh.model.occ.addCurveLoop(1:4)
gmsh.model.occ.addPlaneSurface([1])
gmsh.model.occ.synchronize()

# set resolution
h = 0.02
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
# gmsh.model.mesh.setSize([(0, 1), (0, 2)], h/10)

# periodic boundary condition
translation = [1, 0, 0, 1, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, 1]
# gmsh.model.mesh.setPeriodic(0, [2, 3], [1, 4], translation)
gmsh.model.mesh.setPeriodic(1, [2], [4], translation)
gmsh.model.occ.synchronize()

# define bottom, surface, and interior
gmsh.model.addPhysicalGroup(0, 1:2, 1, "bot")
gmsh.model.addPhysicalGroup(0, 3:4, 2, "sfc")
gmsh.model.addPhysicalGroup(1, [1], 1, "bot")
gmsh.model.addPhysicalGroup(1, [3], 2, "sfc")
gmsh.model.addPhysicalGroup(1, [2, 4], 3, "int")
gmsh.model.addPhysicalGroup(2, [1], 3, "int")

gmsh.model.mesh.generate(2)
gmsh.write("periodic_square.msh")
gmsh.finalize()