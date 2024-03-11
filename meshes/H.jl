include("mesh_making_utils.jl")

function generate_mesh(h)
    # init
    gmsh.initialize()
    
    # model
    gmsh.model.add("mesh")

    # points
        gmsh.model.occ.addPoint(0.0, 0.0, 0, h) 
        gmsh.model.occ.addPoint(0.0, 1.0, 0, h) 
        gmsh.model.occ.addPoint(0.2, 1.0, 0, h) 
        gmsh.model.occ.addPoint(0.2, 0.6, 0, h) 
        gmsh.model.occ.addPoint(0.5, 0.6, 0, h) 
        gmsh.model.occ.addPoint(0.5, 1.0, 0, h) 
        gmsh.model.occ.addPoint(0.7, 1.0, 0, h) 
        gmsh.model.occ.addPoint(0.7, 0.0, 0, h) 
        gmsh.model.occ.addPoint(0.5, 0.0, 0, h) 
        gmsh.model.occ.addPoint(0.5, 0.4, 0, h) 
        gmsh.model.occ.addPoint(0.2, 0.4, 0, h) 
    N = gmsh.model.occ.addPoint(0.2, 0.0, 0, h) 
    

    # lines
    for i ∈ 1:N-1
        gmsh.model.occ.addLine(i, i+1)
    end
    gmsh.model.occ.addLine(N, 1)
    
    # curves
    gmsh.model.occ.addCurveLoop(1:N)

    # surfaces
    gmsh.model.occ.addPlaneSurface([1])

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

make_meshes(generate_mesh, :H)