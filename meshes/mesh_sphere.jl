using Gmsh: gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("sphere")

gmsh.model.occ.addSphere(0, 0, 0, 1)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(0, 1:2, 1, "boundary")
gmsh.model.addPhysicalGroup(1, [2], 1, "boundary")
gmsh.model.addPhysicalGroup(2, [1], 1, "boundary")
gmsh.model.addPhysicalGroup(3, [1], 2, "interior")

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
gmsh.model.mesh.generate(3)
gmsh.write(joinpath(@__DIR__, "sphere.msh"))
gmsh.finalize()
