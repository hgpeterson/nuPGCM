using HDF5
import Gmsh: gmsh

function generate_bowl_mesh(h₀)
    # init
    gmsh.initialize()
    
    # model
    gmsh.model.add("bowl_mesh")

    # volumes
    gmsh.model.occ.addSphere(0, 0, 1, √2, 1) 
    gmsh.model.occ.addBox(-2, -2, 0, 4, 4, 4, 2)

    # cut cube out of sphere
    gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3)

    # sync 
    gmsh.model.occ.synchronize()

    # physical groups
    gmsh.model.addPhysicalGroup(1, [1, 2, 3], 1, "coastline")
    gmsh.model.addPhysicalGroup(2, [1], 2, "bottom")
    gmsh.model.addPhysicalGroup(2, [2], 3, "surface")
    gmsh.model.addPhysicalGroup(3, [3], 4, "interior")

    # mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h₀)

    # generate mesh
    gmsh.model.mesh.generate(3)
    
    # save
    gmsh.write("mesh.msh")
    gmsh.finalize()
end

h₀ = 0.16
generate_bowl_mesh(h₀)