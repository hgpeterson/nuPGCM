using Gmsh: gmsh
using Printf

"""
    mesh_periodic_box(h, α; W=1.0, L=1.0)

Generate a box `[0, W] × [-L/2, L/2] × [-H, 0]` with `H = α*W`, periodic in x.

Physical groups: "bottom" (z = -H only), "surface" (z = 0), "channel_west" (x = 0),
"channel_east" (x = W), "wall" (y = ±L/2), and "interior".
"""
function mesh_periodic_box(h, α; W=1.0, L=1.0)
    H = α*W

    gmsh.initialize()
    gmsh.model.add("periodic_box")

    gmsh.model.occ.addBox(0, -L/2, -H, W, L, H)
    gmsh.model.occ.synchronize()

    # classify boundary surfaces by center of mass
    s_bot = s_sfc = s_west = s_east = 0
    s_wall = Int[]
    for (dim, tag) in gmsh.model.getEntities(2)
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if isapprox(com[3], -H; atol=1e-8)
            s_bot = tag
        elseif isapprox(com[3], 0; atol=1e-8)
            s_sfc = tag
        elseif isapprox(com[1], 0; atol=1e-8)
            s_west = tag
        elseif isapprox(com[1], W; atol=1e-8)
            s_east = tag
        else
            push!(s_wall, tag)
        end
    end

    # make east wall periodic image of west wall
    translation = [1, 0, 0, W,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [s_east], [s_west], translation)

    gmsh.model.addPhysicalGroup(2, [s_bot],  1, "bottom")
    gmsh.model.addPhysicalGroup(2, [s_sfc],  2, "surface")
    gmsh.model.addPhysicalGroup(2, [s_west], 3, "channel_west")
    gmsh.model.addPhysicalGroup(2, [s_east], 4, "channel_east")
    gmsh.model.addPhysicalGroup(2, s_wall,   5, "wall")
    gmsh.model.addPhysicalGroup(3, [1],      6, "interior")

    # set resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    gmsh.model.mesh.generate(3)
    gmsh.write(joinpath(@__DIR__, @sprintf("periodic_box_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end
