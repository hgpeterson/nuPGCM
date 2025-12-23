using Gmsh: gmsh

gmsh.initialize()
gmsh.model.add("turtle")

# parameters
h = 0.1
α = 1/2
R = (1 + α^2) / (2α)
c = R - α

# body parts as spheres
shell_tag = gmsh.model.occ.addSphere(0, 0, c, R)
leg1_tag = gmsh.model.occ.addSphere(+√2/2, +√2/2, 0, 0.3)
leg2_tag = gmsh.model.occ.addSphere(-√2/2, +√2/2, 0, 0.3)
leg3_tag = gmsh.model.occ.addSphere(+√2/2, -√2/2, 0, 0.3)
leg4_tag = gmsh.model.occ.addSphere(-√2/2, -√2/2, 0, 0.3)
tail_tag = gmsh.model.occ.addCone(0, -0.8, 0.0, 
                                  0, -0.5, 0.0, 
                                  0.2, 0)
head_tag = gmsh.model.occ.addSphere(0, 1.1, 0, 0.4)

# combine body parts
body_tag = head_tag + 1
gmsh.model.occ.fuse([(3, shell_tag)], [(3, leg1_tag)], body_tag, true, true); body_tag += 1
gmsh.model.occ.fuse([(3, body_tag-1)], [(3, leg2_tag)], body_tag, true, true); body_tag += 1
gmsh.model.occ.fuse([(3, body_tag-1)], [(3, leg3_tag)], body_tag, true, true); body_tag += 1
gmsh.model.occ.fuse([(3, body_tag-1)], [(3, leg4_tag)], body_tag, true, true); body_tag += 1
gmsh.model.occ.fuse([(3, body_tag-1)], [(3, tail_tag)], body_tag, true, true); body_tag += 1
gmsh.model.occ.fuse([(3, body_tag-1)], [(3, head_tag)], body_tag, true, true)
gmsh.model.occ.synchronize()

# box to slice off the top
box_tag = gmsh.model.occ.addBox(-2, -2, 0, 4, 4, 2)

# cut the top off
a = gmsh.model.occ.cut([(3, body_tag)], [(3, box_tag)], true, true)
gmsh.model.occ.synchronize()

# Physical groups
gmsh.model.addPhysicalGroup(0, [2], 1, "sfc")
gmsh.model.addPhysicalGroup(0, vcat(1, 3:41), 2, "bot")
gmsh.model.addPhysicalGroup(1, [1, 3, 5, 8, 11, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], 1, "sfc")
gmsh.model.addPhysicalGroup(1, [2, 4, 6, 7, 9, 10, 12, 13, 15, 17, 29, 31, 33, 35, 38, 40], 2, "bot")
gmsh.model.addPhysicalGroup(2, [2], 1, "sfc")
gmsh.model.addPhysicalGroup(2, [1, 3, 4, 5, 6, 7, 8, 9], 2, "bot")
gmsh.model.addPhysicalGroup(3, [1], 3, "int")

# set mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Save mesh to file
# gmsh.write("turtle.msh")
gmsh.write("turtle.vtk")

gmsh.finalize()