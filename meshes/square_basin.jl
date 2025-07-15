using Gmsh: gmsh

gmsh.initialize()
gmsh.model.add("square_basin")

# fake bathymetry on the square [0, L] x [0, L]
L = 1
h = 0.1
x = 0:h:L
nx = length(x)
y = 0:h:L
ny = length(y)
H = 5*[(L - x[i])*x[i]*(L - y[j])*y[j] for i in 1:nx, j in 1:ny] # some simple depth function H(x, y)

# coast_mask[i, j] is 1 if the point is on the coast, 0 otherwise
coast_mask = zeros(Int, nx, ny)  
coast_mask[1, :] .= 1    # left edge
coast_mask[nx, :] .= 1   # right edge
coast_mask[:, 1] .= 1    # bottom edge
coast_mask[:, ny] .= 1   # top edge
H[coast_mask .== 1] .= 0 # ensure H = 0 on coast

# add points at bottom
pmap = zeros(Int, nx, ny) # pmap[i, j] is the tag number of the point at x[i], y[j]
for i in 1:nx, j in 1:ny 
    pmap[i, j] = gmsh.model.geo.addPoint(x[i], y[j], -H[i, j], h)
end

# connect bottom points as patches
l_coast = Int[] # indices of lines along coast
for i in 1:nx-1, j in 1:ny-1
    # 4 lines create a bottom patch
    l1 = gmsh.model.geo.addLine(pmap[i, j], pmap[i + 1, j])
    l2 = gmsh.model.geo.addLine(pmap[i + 1, j], pmap[i + 1, j + 1])
    l3 = gmsh.model.geo.addLine(pmap[i + 1, j + 1], pmap[i, j + 1])
    l4 = gmsh.model.geo.addLine(pmap[i, j + 1], pmap[i, j])

    # add line index if it's along the coast
    (coast_mask[i, j] == 1 && coast_mask[i+1, j] == 1) ? push!(l_coast, l1) : nothing
    (coast_mask[i + 1, j] == 1 && coast_mask[i + 1, j + 1] == 1) ? push!(l_coast, l2) : nothing
    (coast_mask[i, j + 1] == 1 && coast_mask[i + 1, j + 1] == 1) ? push!(l_coast, l3) : nothing
    (coast_mask[i, j] == 1 && coast_mask[i, j + 1] == 1) ? push!(l_coast, l4) : nothing

    # define patch
    c = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addSurfaceFilling([c])
end

# z = 0 surface
c = gmsh.model.geo.addCurveLoop(l_coast)
s = gmsh.model.geo.addPlaneSurface([c])

# generate
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("square_basin.msh")
gmsh.finalize()
