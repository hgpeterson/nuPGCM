using nuPGCM
using HDF5
using PyPlot
import Gmsh: gmsh

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    e = boundary_nodes(t)

Find all boundary nodes `e` in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices = nuPGCM.all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

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

    # # quick square
    # gmsh.model.geo.addPoint(-1, 0, 0, h₀)
    # gmsh.model.geo.addPoint(-1, -1, 0, h₀)
    # gmsh.model.geo.addPoint(1, -1, 0, h₀)
    # gmsh.model.geo.addPoint(1, 0, 0, h₀)
    # N = 4

    # depth function
    z(x) = x^2 - 1
    # ∂ₓz(x) = 2*x
    # z(x) = -1
    
    # edge points
    x = -1:h₀/r:1
    n = size(x, 1)
    for i=1:n
        gmsh.model.geo.addPoint(x[i], z(x[i]), 0, h₀)
    end
    for i=(n-1):-1:2
        gmsh.model.geo.addPoint(x[i], 0, 0, h₀)
    end
    N = 2*n - 2

    # # edge points
    # x = -1
    # N = 0
    # while z(x) <= 0
    #     if x > 0 && z(x) > -h₀/r/2
    #         break
    #     end
    #     gmsh.model.geo.addPoint(x, z(x), 0, h₀)
    #     x += h₀/r/sqrt(1 + ∂ₓz(x)^2)
    #     N += 1
    # end
    # gmsh.model.geo.addPoint(1, 0, 0, h₀)
    # N += 1
    
    # connect edge points by lines
    for i=1:N-1
        gmsh.model.geo.addLine(i, i + 1)
    end
    gmsh.model.geo.addLine(N, 1)

    # for x=-1:0.5:1
    #     gmsh.model.geo.addPoint(x, z(x), 0, h₀)
    # end
    # gmsh.model.geo.addPoint(1, 0, 0, h₀)
    # gmsh.model.geo.addPoint(-1, 0, 0, h₀)
    # gmsh.model.geo.addSpline(1:5, 1)
    # gmsh.model.geo.addLine(5, 6, 2)
    # gmsh.model.geo.addLine(6, 7, 3)
    # gmsh.model.geo.addLine(7, 1, 4)
    # N = 4
    
    # loop curves together and define surface
    gmsh.model.geo.addCurveLoop(1:N, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # # refine mesh near boundary nodes
    # gmsh.model.mesh.field.add("Distance", 1)
    # gmsh.model.mesh.field.setNumbers(1, "CurvesList", 1:N)
    # gmsh.model.mesh.field.setNumber(1, "Sampling", 10)

    # gmsh.model.mesh.field.add("Threshold", 2)
    # gmsh.model.mesh.field.setNumber(2, "InField", 1)
    # gmsh.model.mesh.field.setNumber(2, "SizeMin", h₀/r)
    # gmsh.model.mesh.field.setNumber(2, "SizeMax", h₀)
    # gmsh.model.mesh.field.setNumber(2, "DistMin", 0.01)
    # gmsh.model.mesh.field.setNumber(2, "DistMax", 0.02)

    # gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # # turn off the usual ways the mesh size is determined
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # use different mesh algorithm (better for variable mesh size)
    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    
    # sync and generate
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    # save and show
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
function load_gmesh(; h5save=false)
    # initialize mesh and load from file
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl_mesh")
    gmsh.open("mesh.msh")

    # find triangle nodes from the elements in the type-2 surface with tag 1
    tri_nodes = gmsh.model.mesh.getElements(2, 1)[3][1]
    nt = Int64(size(tri_nodes, 1)/3)
    t = zeros(nt, 3)
    for i=1:nt
        t[i, :] = [tri_nodes[3*i-2] tri_nodes[3*i-1] tri_nodes[3*i]]
    end
    t = Int64.(t)

    # find node positions by looping through indices
    np = Int64(maximum(t))
    p = zeros(np, 2)
    for i=1:np
        p[i, :] = gmsh.model.mesh.getNode(i)[1][1:2]
    end

    # get edge nodes
    e = boundary_nodes(t)

    gmsh.finalize()

    # save as h5
    if h5save
        file = h5open("mesh.h5", "w")
        write(file, "p", p)
        write(file, "t", t)
        write(file, "e", e)
        close(file)
        println("mesh.h5")
    end

    return p, t, e
end

generate_bowl_mesh(0.08, 1)
p, t, e = load_gmesh(h5save=true)
tplot(p, t)
plot(p[e, 1], p[e, 2], "o", ms=1)
plt.axis("equal")
savefig("mesh.png")
println("mesh.png")
plt.close()