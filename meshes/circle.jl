using HDF5
import Gmsh: gmsh

function generate_circle_mesh(h)
    # init
    gmsh.initialize()
    
    # model
    gmsh.model.add("circle")

    # volumes
    gmsh.model.occ.addCircle(0, 0, 0, 1, 1) 
    gmsh.model.occ.addCurveLoop([1], 1)
    gmsh.model.occ.addPlaneSurface([1], 1)

    # sync 
    gmsh.model.occ.synchronize()

    # physical groups
    gmsh.model.geo.addPhysicalGroup(1, [1], 1)
    gmsh.model.geo.addPhysicalGroup(2, [1], 2)

    # mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # generate mesh
    gmsh.model.mesh.generate(2)
    
    # save
    gmsh.write("mesh.msh")
    gmsh.finalize()
end

generate_circle_mesh(0.04)