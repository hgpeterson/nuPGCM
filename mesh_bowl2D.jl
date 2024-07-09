using NonhydroPG
using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function generate_bowl_mesh(h)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl2D")

    # points
    gmsh.model.geo.addPoint(-L, 0, 0, h)
    gmsh.model.geo.addPoint(0, -2, 0, h) # control point
    gmsh.model.geo.addPoint(L, 0, 0, h)

    # lines
    gmsh.model.geo.addBezier([1, 2, 3], 1) # bezier curve through (-1, 0), (0, -2), (1, 0) is a parabola with z = x^2 - 1
    gmsh.model.geo.addLine(3, 1, 2)

    # loop curves together and define surface
    gmsh.model.geo.addCurveLoop(1:2, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    # sync 
    gmsh.model.geo.synchronize()

    # define boundary and interior physical groups
    gmsh.model.addPhysicalGroup(0, [1, 3], 1, "bot")
    gmsh.model.addPhysicalGroup(1, [1], 1, "bot")
    gmsh.model.addPhysicalGroup(1, [2], 2, "sfc")
    gmsh.model.addPhysicalGroup(2, [1], 3, "int")

    # # refine near boundary
    # rf = 2
    # Δ = 0.05
    # gmsh.model.mesh.field.add("Distance", 1)
    # gmsh.model.mesh.field.setNumbers(1, "CurvesList", [d[2] for d in gmsh.model.getEntities(1)])
    # gmsh.model.mesh.field.setNumber(1, "Sampling", rf/h)
    # # gmsh.model.mesh.field.add("Threshold", 2)
    # # gmsh.model.mesh.field.setNumber(2, "InField", 1)
    # # gmsh.model.mesh.field.setNumber(2, "SizeMin", h/4)
    # # gmsh.model.mesh.field.setNumber(2, "SizeMax", h)
    # # gmsh.model.mesh.field.setNumber(2, "DistMin", h)
    # # gmsh.model.mesh.field.setNumber(2, "DistMax", 4h)
    # gmsh.model.mesh.field.add("MathEval", 2)
    # gmsh.model.mesh.field.setString(2, "F", "$h*(1 - $((rf - 1)/rf)*exp(-F1/$Δ))")
    # gmsh.model.mesh.field.setAsBackgroundMesh(2)
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # gmsh.option.setNumber("Mesh.Algorithm", 5)

    # generate mesh
    gmsh.model.mesh.generate(2)

    # save
    gmsh.write("bowl2D.msh")
    gmsh.finalize()
end

function mesh_plot(p, t)
    fig, ax = plt.subplots(1)
    ax.tripcolor(p[:, 1], p[:, 2], t .- 1, 0*t[:, 1], cmap="Greys", edgecolors="k", lw=0.2, rasterized=true)
    # ax.plot(-1:0.01:1, (-1:0.01:1).^2 .- 1, "r")
    ax.axis("equal")
    # ax.set_xlim(-L, -L+1.0)
    # ax.set_ylim(-0.1, 0)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig("images/bowl2D.png")
    println("images/bowl2D.png")
    plt.close()
end

L = 1
generate_bowl_mesh(0.02)
p, t = get_p_t("bowl2D.msh")
mesh_plot(p, t)
