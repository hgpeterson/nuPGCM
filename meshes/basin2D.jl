using Gmsh: gmsh

h = 0.1 # resolution

gmsh.initialize()
gmsh.model.add("basin2D")

# surface "mesh"
x = -1:0.01:1
nx = length(x)

# depth function
H = 1 .- x.^2

# add points at z = -H
for i in 1:nx
    gmsh.model.geo.addPoint(x[i], 0, -H[i], h)
end

# connect as lines
for i in 1:nx-1
    gmsh.model.geo.addLine(i, i+1)
end
gmsh.model.geo.addLine(nx, 1)

# loop together to define region
gmsh.model.geo.addCurveLoop(1:nx)
gmsh.model.geo.addPlaneSurface([1], 1)

# generate
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("basin2D.msh")
gmsh.finalize()
