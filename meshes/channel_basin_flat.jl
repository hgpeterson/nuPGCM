using Gmsh: gmsh
using Printf

function mesh_channel_basin_flat(h, α)
    # params
    L = 2
    W = 1
    L_channel = L/4
    y0 = -L/2 + L_channel # channel/basin cutoff
    H = α*W

    gmsh.initialize()

    gmsh.model.add("channel_basin_flat")

    # create points and store them in an array that reflects the geometry of the mesh
    p = zeros(Int, 2, 3, 2)
    p[1, 1, 1] = gmsh.model.occ.addPoint(0, -L/2, -H)
    p[1, 2, 1] = gmsh.model.occ.addPoint(0,   y0, -H)
    p[1, 3, 1] = gmsh.model.occ.addPoint(0,  L/2, -H)
    p[1, 1, 2] = gmsh.model.occ.addPoint(0, -L/2,  0)
    p[1, 2, 2] = gmsh.model.occ.addPoint(0,   y0,  0)
    p[1, 3, 2] = gmsh.model.occ.addPoint(0,  L/2,  0)
    p[2, 1, 1] = gmsh.model.occ.addPoint(W, -L/2, -H)
    p[2, 2, 1] = gmsh.model.occ.addPoint(W,   y0, -H)
    p[2, 3, 1] = gmsh.model.occ.addPoint(W,  L/2, -H)
    p[2, 1, 2] = gmsh.model.occ.addPoint(W, -L/2,  0)
    p[2, 2, 2] = gmsh.model.occ.addPoint(W,   y0,  0)
    p[2, 3, 2] = gmsh.model.occ.addPoint(W,  L/2,  0)

    # connect points to create lines
    l = zeros(Int, p[end, end, end], p[end, end, end]) # l[i, j] is the line connecting points i and j 
    function add_line!(l, i, j)
        l[i, j] = gmsh.model.occ.addLine(i, j)
        l[j, i] = -l[i, j]
        return l[i, j]
    end
    l_bot = zeros(Int, 12)
    l_sfc = zeros(Int, 6)
    # bottom loop
    l_bot[1] = add_line!(l, p[1, 1, 1], p[1, 2, 1])
    l_bot[2] = add_line!(l, p[1, 2, 1], p[1, 3, 1])
    l_bot[3] = add_line!(l, p[1, 3, 1], p[2, 3, 1])
    l_bot[4] = add_line!(l, p[2, 3, 1], p[2, 2, 1])
    l_bot[5] = add_line!(l, p[2, 2, 1], p[2, 1, 1])
    l_bot[6] = add_line!(l, p[2, 1, 1], p[1, 1, 1])
    # top loop
    l_sfc[1] = add_line!(l, p[1, 1, 2], p[1, 2, 2])
    l_sfc[2] = add_line!(l, p[1, 2, 2], p[1, 3, 2])
    l_sfc[3] = add_line!(l, p[1, 3, 2], p[2, 3, 2])
    l_sfc[4] = add_line!(l, p[2, 3, 2], p[2, 2, 2])
    l_sfc[5] = add_line!(l, p[2, 2, 2], p[2, 1, 2])
    l_sfc[6] = add_line!(l, p[2, 1, 2], p[1, 1, 2])
    # connecting lines
    l_bot[7]  = add_line!(l, p[1, 1, 1], p[1, 1, 2])
    l_bot[8]  = add_line!(l, p[1, 2, 1], p[1, 2, 2])
    l_bot[9]  = add_line!(l, p[1, 3, 1], p[1, 3, 2])
    l_bot[10] = add_line!(l, p[2, 1, 1], p[2, 1, 2])
    l_bot[11] = add_line!(l, p[2, 2, 1], p[2, 2, 2])
    l_bot[12] = add_line!(l, p[2, 3, 1], p[2, 3, 2])

    # connect lines to create surfaces
    c = gmsh.model.occ.addCurveLoop([l[p[1, 1, 1], p[1, 2, 1]], 
                                    l[p[1, 2, 1], p[1, 3, 1]], 
                                    l[p[1, 3, 1], p[2, 3, 1]], 
                                    l[p[2, 3, 1], p[2, 2, 1]], 
                                    l[p[2, 2, 1], p[2, 1, 1]], 
                                    l[p[2, 1, 1], p[1, 1, 1]]]) # bottom
    s_bot = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[1, 1, 2], p[1, 2, 2]], 
                                    l[p[1, 2, 2], p[1, 3, 2]], 
                                    l[p[1, 3, 2], p[2, 3, 2]], 
                                    l[p[2, 3, 2], p[2, 2, 2]], 
                                    l[p[2, 2, 2], p[2, 1, 2]], 
                                    l[p[2, 1, 2], p[1, 1, 2]]]) # surface
    s_sfc = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[1, 1, 1], p[1, 1, 2]], 
                                    l[p[1, 1, 2], p[2, 1, 2]], 
                                    l[p[2, 1, 2], p[2, 1, 1]], 
                                    l[p[2, 1, 1], p[1, 1, 1]]]) # south 
    s_south = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[1, 3, 1], p[1, 3, 2]], 
                                    l[p[1, 3, 2], p[2, 3, 2]], 
                                    l[p[2, 3, 2], p[2, 3, 1]], 
                                    l[p[2, 3, 1], p[1, 3, 1]]]) # north 
    s_north = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[1, 1, 1], p[1, 2, 1]], 
                                    l[p[1, 2, 1], p[1, 2, 2]], 
                                    l[p[1, 2, 2], p[1, 1, 2]], 
                                    l[p[1, 1, 2], p[1, 1, 1]]]) # channel_west 
    s_channel_west = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[2, 1, 1], p[2, 2, 1]], 
                                    l[p[2, 2, 1], p[2, 2, 2]], 
                                    l[p[2, 2, 2], p[2, 1, 2]],
                                    l[p[2, 1, 2], p[2, 1, 1]]]) # channel_east
    s_channel_east = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[1, 2, 1], p[1, 3, 1]], 
                                    l[p[1, 3, 1], p[1, 3, 2]], 
                                    l[p[1, 3, 2], p[1, 2, 2]], 
                                    l[p[1, 2, 2], p[1, 2, 1]]]) # basin_west 
    s_basin_west = gmsh.model.occ.addPlaneSurface([c])
    c = gmsh.model.occ.addCurveLoop([l[p[2, 2, 1], p[2, 3, 1]], 
                                    l[p[2, 3, 1], p[2, 3, 2]], 
                                    l[p[2, 3, 2], p[2, 2, 2]],
                                    l[p[2, 2, 2], p[2, 2, 1]]]) # basin_east
    s_basin_east = gmsh.model.occ.addPlaneSurface([c])

    # connect surfaces to create volume
    gmsh.model.occ.addSurfaceLoop([s_bot, s_sfc, s_south, s_north, 
                                s_channel_west, s_channel_east, 
                                s_basin_west, s_basin_east])
    gmsh.model.occ.addVolume([1])

    # make channel periodic
    gmsh.model.occ.synchronize()
    translation = [1, 0, 0, W, 
                0, 1, 0, 0, 
                0, 0, 1, 0, 
                0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [s_channel_east], [s_channel_west], translation)
    gmsh.model.occ.synchronize()

    # define bottom, surface, coastline, and interior
    gmsh.model.addPhysicalGroup(0, p[:, :, 1][:], 1, "bottom")
    gmsh.model.addPhysicalGroup(0, p[:, :, 2][:], 3, "coastline")
    gmsh.model.addPhysicalGroup(1, l_bot, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, l_sfc[[1, 5]], 2, "surface")
    gmsh.model.addPhysicalGroup(1, l_sfc[[2, 3, 4, 6]], 3, "coastline")
    gmsh.model.addPhysicalGroup(2, [s_bot, s_basin_east, 
                                    s_basin_west, s_south, s_north], 1, "bottom")
    gmsh.model.addPhysicalGroup(2, [s_sfc], 2, "surface")
    gmsh.model.addPhysicalGroup(2, [s_channel_east, s_channel_west], 4, "interior")
    gmsh.model.addPhysicalGroup(3, [1], 4, "interior")

    # set resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    gmsh.model.mesh.generate(3)
    gmsh.write(joinpath(@__DIR__, @sprintf("channel_basin_flat_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end

# h = 0.02
# α = 1/8 # H/W
# mesh_channel_basin_flat(h, α)
# @info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))