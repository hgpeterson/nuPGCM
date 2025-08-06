using Gmsh: gmsh
using Printf

function mesh_basin_flat(h, α)
    gmsh.initialize()

    gmsh.model.add("basin_flat")

    # params
    L = 2
    W = 1
    H = α*W

    gmsh.model.occ.addBox(0, -L/2, -H, 
                        W,    L,  H)
    gmsh.model.occ.synchronize()

    # # define bottom, surface, and interior
    gmsh.model.addPhysicalGroup(0, [2, 4, 6, 8], 1, "bottom")
    gmsh.model.addPhysicalGroup(0, [1, 3, 5, 7], 3, "coastline")
    gmsh.model.addPhysicalGroup(1, [1, 3, 4, 5, 7, 8, 9, 11], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [2, 6, 10, 12], 3, "coastline")
    gmsh.model.addPhysicalGroup(2, 1:5, 1, "bottom")
    gmsh.model.addPhysicalGroup(2, [6], 2, "surface")
    gmsh.model.addPhysicalGroup(3, [1], 4, "interior")

    # set resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    gmsh.model.mesh.generate(3)
    gmsh.write(joinpath(@__DIR__, "basin_flat.msh"))
    gmsh.finalize()
end

h = 0.08
α = 1/2 # H/W
mesh_basin_flat(h, α)
@info @sprintf("2εₘᵢₙ = 2h/(α√2) = %1.1e\n", 2h/(α√2))