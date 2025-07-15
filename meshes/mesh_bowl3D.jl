using Gmsh: gmsh
using Printf

function generate_bowl_mesh_3D(h, α)
    # init
    gmsh.initialize()

    # model
    gmsh.model.add("bowl3D")

    # points
    gmsh.model.occ.addPoint(0, 0, -α, h, 1)
    gmsh.model.occ.addPoint(0.5, 0, -α, h, 2) # control point
    gmsh.model.occ.addPoint(1, 0, 0, h, 3)

    # curve z = α*(1 - x^2)
    gmsh.model.occ.addBezier([1, 2, 3], 1)

    # revolve curve around z-axis (makes surface 1)
    gmsh.model.occ.revolve([(1, 1)], 0, 0, -1, 0, 0, 1, 2π)

    # define curve loop made by revolving point 3 around z-axis as curve 2 
    gmsh.model.occ.addCurveLoop([3], 2)

    # fill in the circle made by curve 2 (surface 2)
    gmsh.model.occ.addSurfaceFilling(2, 2)

    # surfaces 1 and 2 make up the bowl
    gmsh.model.occ.addSurfaceLoop(1:2, 1)
    gmsh.model.occ.addVolume([1], 1)

    # sync and define physical groups
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(0, [1, 3], 2, "bot")
    gmsh.model.addPhysicalGroup(1, [1, 3], 2, "bot")
    gmsh.model.addPhysicalGroup(2, [2], 1, "sfc")
    gmsh.model.addPhysicalGroup(2, [3], 2, "bot")
    gmsh.model.addPhysicalGroup(3, [1], 3, "int")

    # gmsh.option.setNumber("Mesh.QualityType", 3)
    # gmsh.option.setNumber("Mesh.Smoothing", 10)
    # gmsh.option.setNumber("Mesh.Algorithm3D", 7)
    # gmsh.option.setNumber("Mesh.QualityType", 0)
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    # gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.4)
    # gmsh.option.setNumber("Mesh.AllowSwapAngle", 25)

    # generate and save mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(3)
    # gmsh.model.mesh.optimize("Netgen")
    # gmsh.model.mesh.optimize("HighOrder")
    # gmsh.model.mesh.optimize("HighOrderElastic")
    gmsh.write(@sprintf("bowl3D_%e_%e.msh", h, α))
    gmsh.write(@sprintf("bowl3D_%e_%e.vtk", h, α))
    gmsh.finalize()
end

# # params
# h = 4e-2
# α = 1
# @info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))

# # generate
# generate_bowl_mesh_3D(h, α)