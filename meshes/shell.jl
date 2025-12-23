using Gmsh: gmsh

gmsh.initialize()
gmsh.model.add("spherical_shell")

# parameters
h = 0.05
α = 1/2
R = 1.0
r = 1.0 - α

# Add outer sphere (radius 1.0)
outer = gmsh.model.occ.addSphere(0, 0, 0, R, 1)

# Add inner sphere (radius 1.0 - α)
inner = gmsh.model.occ.addSphere(0, 0, 0, r, 2)

# Subtract inner from outer to make shell
shell = gmsh.model.occ.cut([(3, outer)], [(3, inner)], 3, true, true)

# Synchronize CAD kernel with Gmsh model
gmsh.model.occ.synchronize()

# Add a bump to the inner surface
lat = 30*π/180
lon =  0*π/180
ρ = 0.7*r
bump = gmsh.model.occ.addSphere(ρ*cos(lat)*cos(lon), ρ*cos(lat)*sin(lon), ρ*sin(lat), α/2, 4)
# gmsh.model.occ.fuse([(3, shell[1][1][2])], [(3, bump)], 5, true, true)
gmsh.model.occ.cut([(3, shell[1][1][2])], [(3, bump)], 5, true, true)
gmsh.model.occ.synchronize()

# Physical groups
gmsh.model.addPhysicalGroup(0, [1, 2], 1, "sfc")
gmsh.model.addPhysicalGroup(0, [3, 4, 5, 6, 7], 2, "bot")
gmsh.model.addPhysicalGroup(1, [1, 2, 3], 1, "sfc")
gmsh.model.addPhysicalGroup(1, [4, 5, 6, 7, 8, 9, 10, 11], 2, "bot")
gmsh.model.addPhysicalGroup(2, [1], 1, "sfc")
gmsh.model.addPhysicalGroup(2, [2, 3], 2, "bot")
gmsh.model.addPhysicalGroup(3, [5], 3, "int")

# Set mesh size for all points (optional, since spheres are parametric surfaces)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Save mesh to file
gmsh.write("spherical_shell.msh")

gmsh.finalize()