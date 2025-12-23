using Gmsh: gmsh
using Printf

function mesh_channel_basin(h, α)
    # params
    L = 2
    W = 1
    L_channel = L/4
    L_flat_channel = L_channel/4 # length of flat part of channel
    L_curve_channel = (L_channel - L_flat_channel)/2 # length of each curved part of channel
    y_channel = -L/2 + L_channel/2 # channel center
    W_flat_basin = W/2 # width of flat part of basin
    W_curve_basin = (W - W_flat_basin)/2 # width of each curved part of basin
    L_basin = L/2 - y_channel # length of basin
    L_curve_basin = W_curve_basin # length of curved end of basin
    H = α*W

    # init
    gmsh.initialize()
    gmsh.model.add("channel_basin")

    # channel
    p1 = gmsh.model.occ.addPoint(0, -L/2,                                        0)
    p2 = gmsh.model.occ.addPoint(0, -L/2 + L_curve_channel,                     -H)
    p3 = gmsh.model.occ.addPoint(0, -L/2 + L_curve_channel + L_flat_channel,    -H)
    p4 = gmsh.model.occ.addPoint(0, -L/2 + L_channel,                            0)
    p5 = gmsh.model.occ.addPoint(0, -L/2 + L_curve_channel/2,                   -H) # control point
    p6 = gmsh.model.occ.addPoint(0, -L/2 + 3L_curve_channel/2 + L_flat_channel, -H) # control point
    l1 = gmsh.model.occ.addBezier([p1, p5, p2])
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addBezier([p3, p6, p4])
    l4 = gmsh.model.occ.addLine(p4, p1)
    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.extrude([(2, s)], W, 0, 0)
    gmsh.model.occ.synchronize()

    # basin
    p1 = gmsh.model.occ.addPoint(0,                               y_channel,  0)
    p2 = gmsh.model.occ.addPoint(W_curve_basin,                   y_channel, -H)
    p3 = gmsh.model.occ.addPoint(W_curve_basin + W_flat_basin,    y_channel, -H)
    p4 = gmsh.model.occ.addPoint(W,                               y_channel,  0)
    p5 = gmsh.model.occ.addPoint(W_curve_basin/2,                 y_channel, -H) # control point
    p6 = gmsh.model.occ.addPoint(3W_curve_basin/2 + W_flat_basin, y_channel, -H) # control point
    l1 = gmsh.model.occ.addBezier([p1, p5, p2])
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addBezier([p3, p6, p4])
    l4 = gmsh.model.occ.addLine(p4, p1)
    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.extrude([(2, s)], 0, L_basin - L_curve_basin, 0)
    gmsh.model.occ.synchronize()

    # rounded end of basin
    p1 = gmsh.model.occ.addPoint(W_curve_basin, L/2 - L_curve_basin,   -H)
    p2 = gmsh.model.occ.addPoint(W_curve_basin, L/2,                    0)
    p3 = gmsh.model.occ.addPoint(W_curve_basin, L/2 - L_curve_basin,    0)
    p4 = gmsh.model.occ.addPoint(W_curve_basin, L/2 - L_curve_basin/2, -H) # control point
    l1 = gmsh.model.occ.addBezier([p1, p4, p2])
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)
    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3])
    s = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.extrude([(2, s)], W_flat_basin, 0, 0)
    gmsh.model.occ.synchronize()

    # fuse the volumes
    gmsh.model.occ.fuse([(3, 1)], [(3, 2), (3, 3)])
    gmsh.model.occ.synchronize()

    """
    Create a rounded corner going from (x1, y) to (x2, y) with a parabolic depth.
    """
    function add_rounded_corner(x1, x2, y)
        p1 = gmsh.model.occ.addPoint(x1,          y, -H)
        p2 = gmsh.model.occ.addPoint(x2,          y,  0)
        p3 = gmsh.model.occ.addPoint((x1 + x2)/2, y, -H) # control point
        l1 = gmsh.model.occ.addBezier([p1, p3, p2])
        out = gmsh.model.occ.revolve([(1, l1)], x1, y, -H, 0, 0, H, 2π)
        curves = findall(x->x[1]==1, out)
        surfaces = findall(x->x[1]==2, out)
        s1 = out[surfaces[1]][2]
        c1 = maximum([out[i][2] for i in curves])
        cl = gmsh.model.occ.addCurveLoop([c1])
        s2 = gmsh.model.occ.addSurfaceFilling(cl)
        sl = gmsh.model.occ.addSurfaceLoop([s1, s2])
        gmsh.model.occ.addVolume([sl])
        gmsh.model.occ.remove([(2, s1)]) # extra surface for some reason??
        gmsh.model.occ.synchronize()

        # copy volume 1 for cut
        gmsh.model.occ.copy([(3, 1)])
        gmsh.model.occ.synchronize()
        gmsh.model.occ.cut([(3, 2)], [(3, 3)])
        gmsh.model.occ.synchronize()
        gmsh.model.occ.fuse([(3, 1)], [(3, 2)])
        gmsh.model.occ.synchronize()
    end

    add_rounded_corner(W_curve_basin,                0,  L/2 - L_curve_basin)
    add_rounded_corner(W_curve_basin + W_flat_basin, W,  L/2 - L_curve_basin)

    # periodic boundary condition
    translation = [1, 0, 0, W, 
                   0, 1, 0, 0, 
                   0, 0, 1, 0, 
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [5], [4], translation)
    gmsh.model.occ.synchronize()

    # define bottom, surface, coastline, and interior
    gmsh.model.addPhysicalGroup(0, [68, 69, 76, 77, 78, 79, 80, 81], 1, "bottom")
    gmsh.model.addPhysicalGroup(0, [66, 67, 70, 71, 72, 73, 74, 75], 3, "coastline")
    gmsh.model.addPhysicalGroup(1, vcat([2, 3, 4], 12:28), 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [5, 11], 2, "surface")
    gmsh.model.addPhysicalGroup(1, [1, 6, 7, 8, 9, 10], 3, "coastline")
    gmsh.model.addPhysicalGroup(2, vcat(1, 3, 6:12), 1, "bottom")
    gmsh.model.addPhysicalGroup(2, [2], 2, "surface")
    gmsh.model.addPhysicalGroup(2, [4, 5], 4, "interior")
    gmsh.model.addPhysicalGroup(3, [1], 4, "interior")

    # set resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    gmsh.model.mesh.generate(3)
    gmsh.write(joinpath(@__DIR__, @sprintf("channel_basin_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end

# h = 0.08
# α = 1/2
# mesh_channel_basin(h, α)
# @info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))
