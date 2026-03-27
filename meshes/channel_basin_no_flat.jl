using Gmsh: gmsh
using Printf

function mesh_channel_basin_no_flat(h, α)
    gmsh.initialize()

    gmsh.model.add("channel_basin")

    # params
    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = L_channel/4 # length of flat part of channel
    L_curve_channel = (L_channel - L_flat_channel)/2 # length of each curved part of channel
    y_channel = -L/2 + L_channel/2 # channel center
    L_basin = L/2 - y_channel # length of basin
    H = α*W

    # channel
    p1 = gmsh.model.occ.addPoint(0, -L/2,                                        0)
    p2 = gmsh.model.occ.addPoint(0, -L/2,                                       -H)
    p3 = gmsh.model.occ.addPoint(0, -L/2 + L_curve_channel + L_flat_channel,    -H)
    p4 = gmsh.model.occ.addPoint(0, -L/2 + L_channel,                            0)
    p6 = gmsh.model.occ.addPoint(0, -L/2 + 3L_curve_channel/2 + L_flat_channel, -H) # control point
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addBezier([p3, p6, p4])
    l4 = gmsh.model.occ.addLine(p4, p1)
    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.extrude([(2, s)], W, 0, 0)
    gmsh.model.occ.synchronize()

    # basin
    p1 = gmsh.model.occ.addPoint(0, y_channel, 0)
    p2 = gmsh.model.occ.addPoint(W/2, y_channel, -2α*W) # control point
    p3 = gmsh.model.occ.addPoint(W, y_channel, 0)
    l1 = gmsh.model.occ.addBezier([p1, p2, p3]) # parabola
    l2 = gmsh.model.occ.addLine(p3, p1)
    c1 = gmsh.model.occ.addCurveLoop([l1, l2])
    s1 = gmsh.model.occ.addPlaneSurface([c1])
    # gmsh.model.occ.extrude([(2, s1)], 0, L/2 - W/2 - y_channel, 0)
    gmsh.model.occ.extrude([(2, s1)], 0, L/2 - y_channel, 0)
    gmsh.model.occ.synchronize()

    # # basin curved end
    # p1 = gmsh.model.occ.addPoint(W/2, L/2 - W/2, -H)
    # p2 = gmsh.model.occ.addPoint(0,   L/2 - W/2, 0)
    # p3 = gmsh.model.occ.addPoint(W/4, L/2 - W/2, -H) # control point
    # l1 = gmsh.model.occ.addBezier([p1, p3, p2])
    # out = gmsh.model.occ.revolve([(1, l1)], W/2, L/2 - W/2, -H, 0, 0, H, 2π)
    # cl = gmsh.model.occ.addCurveLoop([out[4][2]])
    # sf = gmsh.model.occ.addSurfaceFilling(cl)
    # sl = gmsh.model.occ.addSurfaceLoop([out[2][2], sf])
    # gmsh.model.occ.addVolume([sl])
    # gmsh.model.occ.synchronize()

    # fuse the three volumes
    gmsh.model.occ.fuse([(3, 1)], [(3, 2)])
    gmsh.model.occ.synchronize()

    # periodic boundary condition
    translation = [1, 0, 0, W, 
                   0, 1, 0, 0, 
                   0, 0, 1, 0, 
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [5], [4], translation)
    gmsh.model.occ.synchronize()

    # # define bottom, surface, coastline, and interior
    gmsh.model.addPhysicalGroup(0, [14, 15, 20, 21, 22, 23], 1, "bottom")
    gmsh.model.addPhysicalGroup(0, [12, 13, 16, 17, 18, 19], 3, "coastline")
    gmsh.model.addPhysicalGroup(1, [2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [5, 9], 2, "surface")
    gmsh.model.addPhysicalGroup(1, [1, 6, 7, 8], 3, "coastline")
    gmsh.model.addPhysicalGroup(2, [1, 3, 6, 7, 8, 9], 1, "bottom")
    gmsh.model.addPhysicalGroup(2, [2], 2, "surface")
    gmsh.model.addPhysicalGroup(2, [4, 5], 4, "interior")
    gmsh.model.addPhysicalGroup(3, [1], 4, "interior")

    # set resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    gmsh.model.mesh.generate(3)
    gmsh.write(joinpath(@__DIR__, @sprintf("channel_basin_no_flat_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end

# mesh_channel_basin_no_flat(0.02, 0.25)