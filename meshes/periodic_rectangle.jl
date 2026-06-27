using Gmsh: gmsh

function mesh_periodic_rectangle(h, α; y=-0.75, W=1)
    H = α*W
    gmsh.initialize()

    gmsh.model.add("periodic_rectangle")

    gmsh.model.occ.addPoint(0, y, -H)
    gmsh.model.occ.addPoint(W, y, -H)
    gmsh.model.occ.addPoint(W, y, 0)
    gmsh.model.occ.addPoint(0, y, 0)
    gmsh.model.occ.addLine(1, 2)
    gmsh.model.occ.addLine(2, 3)
    gmsh.model.occ.addLine(3, 4)
    gmsh.model.occ.addLine(4, 1)
    gmsh.model.occ.addCurveLoop(1:4)
    gmsh.model.occ.addPlaneSurface([1])
    gmsh.model.occ.synchronize()

    # set resolution
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # periodic boundary condition
    translation = [1, 0, 0, 1, 
                   0, 1, 0, 0, 
                   0, 0, 1, 0, 
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, [2], [4], translation)
    gmsh.model.occ.synchronize()

    # define bottom, surface, and interior
    gmsh.model.addPhysicalGroup(0, 1:2, 1, "bottom")
    gmsh.model.addPhysicalGroup(0, 3:4, 2, "surface")
    gmsh.model.addPhysicalGroup(1, [1], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [3], 2, "surface")
    gmsh.model.addPhysicalGroup(1, [2, 4], 3, "interior")
    gmsh.model.addPhysicalGroup(2, [1], 3, "interior")

    gmsh.model.mesh.generate(2)
    gmsh.write(joinpath(@__DIR__, @sprintf("periodic_rectangle_h%.2e_a%.2e.msh", h, α)))
    gmsh.finalize()
end