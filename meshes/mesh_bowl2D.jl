using Gmsh: gmsh
using Printf

function generate_bowl_mesh_2D(h, α)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl2D")

    # setup domain with H = α*(1 - x^2)
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

    # generate and save
    gmsh.model.mesh.generate(2)
    gmsh.write(joinpath(@__DIR__, @sprintf("bowl2D_%e_%e.msh", h, α)))
    gmsh.finalize()
end

# params
h = 2e-2
α = 1/2
@info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))

# generate
generate_bowl_mesh_2D(h, α)