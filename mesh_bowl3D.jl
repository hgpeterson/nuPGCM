using Gmsh: gmsh

function generate_bowl_mesh(h)
    # init
    gmsh.initialize()
    
    # model
    gmsh.model.add("bowl3D")

    # volumes
    gmsh.model.occ.addSphere(0, 0, 1, √2, 1) 
    gmsh.model.occ.addBox(-2, -2, 0, 4, 4, 4, 2)

    # cut cube out of sphere
    gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3)

    # sync 
    gmsh.model.occ.synchronize()

    # bottom
    gmsh.model.addPhysicalGroup(0, 1:2, 1, "bot")
    gmsh.model.addPhysicalGroup(1, 1:2, 1, "bot")
    gmsh.model.addPhysicalGroup(2, [1], 1, "bot")
    # surface
    gmsh.model.addPhysicalGroup(2, [2], 2, "sfc")
    # interior
    gmsh.model.addPhysicalGroup(3, [3], 3, "int")

    # mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # generate mesh
    gmsh.model.mesh.generate(3)
    
    # save
    gmsh.write("bowl3D.msh")
    gmsh.finalize()
end

generate_bowl_mesh(0.02)