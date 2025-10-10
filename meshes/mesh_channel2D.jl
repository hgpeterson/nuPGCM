using Gmsh: gmsh
using Printf

function generate_channel_mesh_2D(h, α)
    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = L_channel/4
    L_curve_channel = (L_channel - L_flat_channel)/2
    H = α*W

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("channel2D")

    gmsh.model.geo.addPoint(0, -L/2,                                         0)
    gmsh.model.geo.addPoint(0, -L/2 +  L_curve_channel/2,                   -H) # control point
    gmsh.model.geo.addPoint(0, -L/2 +  L_curve_channel,                     -H)
    gmsh.model.geo.addPoint(0, -L/2 + 2L_curve_channel + L_flat_channel,    -H)
    gmsh.model.geo.addPoint(0, -L/2 + 2L_curve_channel + L_flat_channel,     0)
    gmsh.model.geo.addBezier([1, 2, 3])
    gmsh.model.geo.addLine(3, 4)
    gmsh.model.geo.addLine(4, 5)
    gmsh.model.geo.addLine(5, 1)
    gmsh.model.geo.addCurveLoop(1:4)
    gmsh.model.geo.addPlaneSurface([1])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(0, [3], 1, "bottom")
    gmsh.model.addPhysicalGroup(0, [1], 3, "coastline")
    gmsh.model.addPhysicalGroup(0, [4], 6, "basin bottom")
    gmsh.model.addPhysicalGroup(0, [5], 5, "basin top")
    gmsh.model.addPhysicalGroup(1, [1, 2], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [4], 2, "surface")
    gmsh.model.addPhysicalGroup(1, [3], 5, "basin")
    gmsh.model.addPhysicalGroup(2, [1], 4, "interior")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)
    gmsh.write(joinpath(@__DIR__, @sprintf("channel2D_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end

h = 1e-2
α = 1/2
@info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))
generate_channel_mesh_2D(h, α)