using nuPGCM
using HDF5
using PyPlot
import Gmsh: gmsh

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    generateMesh(h₀, r)

Creates 2D mesh of bowl with characteristic length `h₀` and boundary refinement
factor `r`. Data saved to "mesh.msh".
"""
function generate_bowl_mesh(h₀, r)
    # init
    gmsh.initialize()
    
    # log
    gmsh.option.setNumber("General.Terminal", 1)
    
    # model
    gmsh.model.add("bowl_mesh")

    # points
    # gmsh.model.geo.addPoint(-1, 0, 0, h₀/r)
    # gmsh.model.geo.addPoint(0, 1-√2, 0, h₀/r)
    # gmsh.model.geo.addPoint(1, 0, 0, h₀/r)
    # gmsh.model.geo.addPoint(0, 1, 0, h₀/r) # center of circle
    H(x) = 1 - x^2
    Hx_max = 2
    dx = h₀/sqrt(1 + Hx_max^2)
    nx = Int64(round(2/dx)) + 1
    x = range(-1, 1, length=nx)
    # L = sqrt(5) + asinh(2)/2
    for i=1:nx-1
        gmsh.model.geo.addPoint(x[i], -H(x[i]), 0, h₀)
    end
    N = 0
    for i=nx:-1:2
        N = gmsh.model.geo.addPoint(x[i], 0, 0, h₀)
    end

    # lines
    # gmsh.model.geo.addCircleArc(1, 4, 2)
    # gmsh.model.geo.addCircleArc(2, 4, 3)
    # gmsh.model.geo.addLine(3, 1)
    for i=1:N
        gmsh.model.geo.addLine(i, mod1(i+1, N))
    end

    # loop curves together and define surface
    gmsh.model.geo.addCurveLoop(1:N, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # # refine mesh near boundary nodes
    # gmsh.model.mesh.field.add("Distance", 1)
    # gmsh.model.mesh.field.setNumbers(1, "CurvesList", 1:3)
    # gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    # gmsh.model.mesh.field.add("MathEval", 2)
    # gmsh.model.mesh.field.setString(2, "F", string(h₀/r, "+", (h₀ - h₀/r)/2, "*(Tanh(10*(F1 - 0.1)) + 1)"))

    # gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # # turn off the usual ways the mesh size is determined
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # # use different mesh algorithm (better for variable mesh size)
    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    
    # sync 
    gmsh.model.geo.synchronize()

    # define boundary and interior physical groups
    # gmsh.model.addPhysicalGroup(0, 1:3, 1, "bottom")
    # gmsh.model.addPhysicalGroup(1, 1:2, 1, "bottom")
    # gmsh.model.addPhysicalGroup(1, [3], 2, "surface")
    # gmsh.model.addPhysicalGroup(2, [1], 1, "face")
    gmsh.model.addPhysicalGroup(0, 1:N, 1, "boundary")
    gmsh.model.addPhysicalGroup(2, [1], 1, "face")

    # generate mesh
    gmsh.model.mesh.generate(2)

    # optimize mesh
    # gmsh.model.mesh.optimize("HighOrder")
    # gmsh.model.mesh.optimize("Laplace2D")
    # gmsh.model.mesh.optimize("Relocate2D")
    # gmsh.model.mesh.optimize("UntangleMeshGeometry")
    
    # save
    gmsh.write("mesh.msh")
    gmsh.finalize()
end

function load_msh(ifile)
    # initialize mesh and load from file
    gmsh.initialize()
    gmsh.open(ifile)

    # get all node positions (`pts[i, 1:2]` are the coordinates of point `i`)
    p = Array(reshape(gmsh.model.mesh.getNodes()[2], (3, :))')[:, 1:2]

    # get triangles
    t = get_elements(2, 1)

    # get boundary nodes
    e = Int64.(gmsh.model.mesh.getNodesForPhysicalGroup(0, 1)[1])

    # finalize
    gmsh.finalize()

    return p, t, e
end

function get_elements(dim, tag)
    # get nodes in elements of dimension `dim` with tag `tag`
    nodes = gmsh.model.mesh.getElements(dim, tag)[3][1]

    # `els[i, j]` is the index of node `j` in element `i`
    els = Array(Int64.(reshape(nodes, (dim+1, :)))')
    return els
end

function msh2h5(ifile, ofile)
    p, t, e = load_msh(ifile)
    file = h5open(ofile, "w")
    write(file, "p", p)
    write(file, "t", t)
    write(file, "e", e)
    close(file)
    println(ofile)
end

# h₀ = 0.1
# r = 1
# generate_bowl_mesh(h₀, r)
# p, t, e = load_msh("mesh.msh")

# ne = size(e, 1)
# ebot = e[2:Int64(ne/2)]
# etop = e[Int64(ne/2)+2:end]
# tplot(p, t)
# plot(p[ebot, 1], p[ebot, 2], "o", ms=1)
# plot(p[etop, 1], p[etop, 2], "o", ms=1)
# axis("equal")
# savefig("mesh.png")
# println("mesh.png")
# plt.close()

for i=0:5
    generate_bowl_mesh(0.16*2.0^(-i), 1)
    msh2h5("mesh.msh", "gmsh_equal_edges/mesh$i.h5")
end