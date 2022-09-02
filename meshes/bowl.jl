using nuPGCM
import Gmsh: gmsh
using PyPlot

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
    generateMesh(lc, bdy_ref)

Creates 2D mesh of bowl with characteristic length `lc` and boundary refinement
factor `bdy_ref`. Data saved to "mesh.msh".
"""
function generate_bowl_mesh(lc, bdy_ref)
    # init
    gmsh.initialize()
    
    # log
    gmsh.option.setNumber("General.Terminal", 1)
    
    # model
    gmsh.model.add("bowl_mesh")

    # depth function
    Δ = 1/5
    G(x) = 1 - exp(-x^2/(2*Δ^2)) 
    H(x) = G(x - 1)*G(x + 1)
    
    # edge points
    pts = []
    push!(pts, gmsh.model.geo.addPoint(-1, 0, 0, bdy_ref*lc))
    x = -1:lc/bdy_ref:1
    n = size(x, 1)
    for i=2:n-1
        push!(pts, gmsh.model.geo.addPoint(x[i], -H(x[i]), 0, bdy_ref*lc))
    end
    push!(pts, gmsh.model.geo.addPoint(1, 0, 0, bdy_ref*lc))
    for i=(n-1):-1:2
        push!(pts, gmsh.model.geo.addPoint(x[i], 0, 0, bdy_ref*lc))
    end
    push!(pts, gmsh.model.geo.addPoint(-1, 0, 0, bdy_ref*lc))
    
    # connect edge points by lines
    curves = []
    for i=1:size(pts, 1)-1
        push!(curves, gmsh.model.geo.addLine(pts[i], pts[i+1]))
    end
    push!(curves, gmsh.model.geo.addLine(pts[end], pts[1]))
    
    # loop curves together and define surface
    gmsh.model.geo.addCurveLoop(curves, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    # sync
    gmsh.model.geo.synchronize()
    
    # # make left bdy copy of right
    # translation = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # gmsh.model.mesh.setPeriodic(1, [curves[1]], [curves[end-1]], translation)
    
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
function load_gmesh()
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

    return p, t, e
end

generate_bowl_mesh(0.1, 4)
p, t, e = load_gmesh()
tplot(p, t)
savefig("mesh.png")
println("mesh.png")