using Gmsh: gmsh

# init
gmsh.initialize()
gmsh.model.add("rect_basin")

# params
h = 0.1
α = 0.5
L = 2
W = 1

# create volume
gmsh.model.occ.addBox(0, -L/2, -α, W, L, α)
gmsh.model.occ.synchronize()

# set resolution
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

# define bottom, surface, and interior
gmsh.model.addPhysicalGroup(0, 1:8, 1, "bot")
gmsh.model.addPhysicalGroup(1, 1:12, 1, "bot")
gmsh.model.addPhysicalGroup(2, 1:5, 1, "bot")
gmsh.model.addPhysicalGroup(2, [6], 2, "sfc")
gmsh.model.addPhysicalGroup(3, [1], 3, "int")

gmsh.model.mesh.generate(3)
gmsh.write("rect_basin.msh")
gmsh.finalize()