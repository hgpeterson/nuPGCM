using NonhydroPG
using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot
using Printf

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function generate_bowl_mesh(h)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl2D")

    # # H = 1 - x^2
    # gmsh.model.geo.addPoint(-1, 0, 0, h)
    # gmsh.model.geo.addPoint(0, -2, 0, h) # control point
    # gmsh.model.geo.addPoint(1, 0, 0, h)
    # gmsh.model.geo.addBezier([1, 2, 3], 1) # bezier curve through (-1, 0), (0, -2), (1, 0) is a parabola with z = x^2 - 1
    # gmsh.model.geo.addLine(3, 1, 2)
    # gmsh.model.geo.addCurveLoop(1:2, 1)
    # gmsh.model.geo.addPlaneSurface([1], 1)
    # gmsh.model.geo.synchronize()
    # gmsh.model.addPhysicalGroup(0, [1, 3], 1, "bot")
    # gmsh.model.addPhysicalGroup(1, [1], 1, "bot")
    # gmsh.model.addPhysicalGroup(1, [2], 2, "sfc")
    # gmsh.model.addPhysicalGroup(2, [1], 3, "int")

    # H = √(2 - x^2) - 1
    gmsh.model.occ.addDisk(0, 1, 0, √2, √2)
    gmsh.model.occ.addRectangle(-4, 0, 0, 8, 8)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.cut([(2, 1)], [(2, 2)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(0, 1:2, 1, "bot")
    gmsh.model.addPhysicalGroup(1, [2], 1, "bot")
    gmsh.model.addPhysicalGroup(1, [1], 2, "sfc")
    gmsh.model.addPhysicalGroup(2, [1], 3, "int")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # generate and save
    gmsh.model.mesh.generate(2)
    gmsh.write(@sprintf("bowl2D_%0.2f.msh", h))
    gmsh.finalize()
end

function mesh_plot(p, t)
    fig, ax = plt.subplots(1)
    ax.tripcolor(p[:, 1], p[:, 2], t .- 1, 0*t[:, 1], cmap="Greys", edgecolors="k", lw=0.2, rasterized=true)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig("bowl2D.png")
    println("bowl2D.png")
    plt.close()
end

h = 0.01
generate_bowl_mesh(h)
p, t = get_p_t(@sprintf("bowl2D_%0.2f.msh", h))
mesh_plot(p, t)
