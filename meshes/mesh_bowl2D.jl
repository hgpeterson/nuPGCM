using nuPGCM
using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot
using Printf

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

function generate_bowl_mesh(h, α)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl2D")

    # H = α*(1 - x^2)
    gmsh.model.geo.addPoint(-1, 0, 0, h)
    gmsh.model.geo.addPoint(0, 0, -2α, h) # control point
    gmsh.model.geo.addPoint(1, 0, 0, h)
    gmsh.model.geo.addBezier([1, 2, 3], 1) # bezier curve through (-1, 0, 0), (0, 0, -2α), (1, 0, 0) is a parabola with z = α*(x^2 - 1)
    gmsh.model.geo.addLine(3, 1, 2)
    gmsh.model.geo.addCurveLoop(1:2, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(0, [1, 3], 1, "bot")
    gmsh.model.addPhysicalGroup(1, [1], 1, "bot")
    gmsh.model.addPhysicalGroup(1, [2], 2, "sfc")
    gmsh.model.addPhysicalGroup(2, [1], 3, "int")

    # # refine near boundary
    # rf = 10
    # Δ = 0.2
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

    # generate and save
    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.optimize("Netgen")
    gmsh.write(@sprintf("bowl2D_%e_%e.msh", h, α))
    gmsh.finalize()
end

function mesh_plot(p, t)
    fig, ax = plt.subplots(1)
    ax.tripcolor(p[:, 1], p[:, 3], t .- 1, 0*t[:, 1], cmap="Greys", edgecolors="k", lw=0.2, rasterized=true)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig("bowl2D.png")
    println("bowl2D.png")
    plt.close()
end

# params
h = 0.007
α = 1/2
@info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))

# generate
generate_bowl_mesh(h, α)

# # plot
# p, t = get_p_t(@sprintf("bowl2D_%e_%e.msh", h, α))
# mesh_plot(p, t)