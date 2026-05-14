using Gmsh: gmsh
using Printf

function generate_channel_mesh_2D(h, α)
    L = 2
    W = 1
    L_channel = L/4
    H = α*W

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("channel2D")

    p1 = gmsh.model.geo.addPoint(0, -L/2,              0)
    p2 = gmsh.model.geo.addPoint(0, -L/2,             -H)
    p3 = gmsh.model.geo.addPoint(0, -L/2 + L_channel, -H)
    p4 = gmsh.model.geo.addPoint(0, -L/2 + L_channel,  0)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([1])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(0, [p1], 3, "coastline")
    gmsh.model.addPhysicalGroup(0, [p2], 1, "bottom")
    gmsh.model.addPhysicalGroup(0, [p3], 6, "basin bottom")
    gmsh.model.addPhysicalGroup(0, [p4], 5, "basin top")
    gmsh.model.addPhysicalGroup(1, [l1, l2], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [l3], 5, "basin")
    gmsh.model.addPhysicalGroup(1, [l4], 2, "surface")
    gmsh.model.addPhysicalGroup(2, [s], 4, "interior")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(2)
    gmsh.write(joinpath(@__DIR__, @sprintf("channel2D_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end