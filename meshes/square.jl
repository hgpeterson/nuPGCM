include("mesh_making_utils.jl")

function generate_rect_mesh(h)
    # init
    gmsh.initialize()
    
    # model
    gmsh.model.add("square")

    # volumes
    gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2, 1) 

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

make_meshes(generate_rect_mesh, :square)