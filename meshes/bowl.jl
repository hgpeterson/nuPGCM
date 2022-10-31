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
    gmsh.model.geo.addPoint(-1, 0, 0, h₀/r)
    gmsh.model.geo.addPoint(0, 1-√2, 0, h₀/r)
    gmsh.model.geo.addPoint(1, 0, 0, h₀/r)
    gmsh.model.geo.addPoint(0, 1, 0, h₀/r) # center of circle

    # lines
    gmsh.model.geo.addCircleArc(1, 4, 2)
    gmsh.model.geo.addCircleArc(2, 4, 3)
    gmsh.model.geo.addLine(3, 1)

    # loop curves together and define surface
    gmsh.model.geo.addCurveLoop(1:3, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # refine mesh near boundary nodes
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", 1:3)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("MathEval", 2)
    gmsh.model.mesh.field.setString(2, "F", string(h₀/r, "+", (h₀ - h₀/r)/2, "*(Tanh(10*(F1 - 0.1)) + 1)"))

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # turn off the usual ways the mesh size is determined
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # use different mesh algorithm (better for variable mesh size)
    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    
    # sync 
    gmsh.model.geo.synchronize()

    # define boundary and interior physical groups
    gmsh.model.addPhysicalGroup(1, [1, 2], 1, "bot")
    gmsh.model.addPhysicalGroup(1, [3], 2, "top")
    gmsh.model.addPhysicalGroup(2, [1], 3, "surface")

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

"""
    p, t, e = load_gmesh()

Loads mesh from "mesh.msh" file and returns `p`, `t`, `e` data structure
where `p` is an n_nodes × n_dims array of node positions, `t` is an
n_tri × 3 array of triangle node indices, and `e` is an n_edge_nodes × 1 array of
edge node indices.
"""
function load_gmesh(; savefile="")
    # initialize mesh and load from file
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl_mesh")
    gmsh.open("mesh.msh")

    # find triangle nodes from the elements in the type-2 surface with tag 1
    el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(2, 1)
    tri_nodes = el_node_tags[1]
    nt = Int64(size(tri_nodes, 1)/3)
    t = zeros(Int64, nt, 3)
    for i=1:nt
        t[i, :] = tri_nodes[3*i-2:3i]
    end

    # find node positions by looping through indices
    np = Int64(maximum(t))
    p = zeros(np, 2)
    for i=1:np
        coords, param_coord, dim, tag = gmsh.model.mesh.getNode(i)
        p[i, :] = coords[1:2]
    end

    # get edge nodes
    e = nuPGCM.boundary_nodes(t)

    gmsh.finalize()

    # save as h5
    if savefile != ""
        file = h5open(savefile, "w")
        write(file, "p", p)
        write(file, "t", t)
        write(file, "e", e)
        close(file)
        println(savefile)
    end

    return p, t, e
end

h₀ = 0.04
r = 4
generate_bowl_mesh(h₀, r)
p, t, e = load_gmesh(savefile="mesh0.h5")
tplot(p, t)
axis("equal")
savefig("mesh.png")
println("mesh.png")