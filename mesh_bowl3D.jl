using Gmsh: gmsh

# for  H = √(2 - x^2 - y^2) - 1

# function generate_bowl_mesh(h)
#     # init
#     gmsh.initialize()
    
#     # model
#     gmsh.model.add("bowl3D")

#     # volumes
#     gmsh.model.occ.addSphere(0, 0, 1, √2, 1) 
#     gmsh.model.occ.addBox(-2, -2, 0, 4, 4, 4, 2)

#     # cut cube out of sphere
#     gmsh.model.occ.cut([(3, 1)], [(3, 2)], 3)

#     # sync 
#     gmsh.model.occ.synchronize()

#     # bottom
#     gmsh.model.addPhysicalGroup(0, 1:2, 1, "bot")
#     gmsh.model.addPhysicalGroup(1, 1:2, 1, "bot")
#     gmsh.model.addPhysicalGroup(2, [1], 1, "bot")
#     # surface
#     gmsh.model.addPhysicalGroup(2, [2], 2, "sfc")
#     # interior
#     gmsh.model.addPhysicalGroup(3, [3], 3, "int")

#     # mesh size
#     gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

#     # generate mesh
#     gmsh.model.mesh.generate(3)
    
#     # save
#     gmsh.write("bowl3D.msh")
#     gmsh.finalize()
# end

# for  H = 1 - x^2 - y^2

function generate_bowl_mesh(h)
    # init
    gmsh.initialize()

    # model
    gmsh.model.add("bowl3D")

    # points
    gmsh.model.occ.addPoint(0, 0, -1, h, 1)
    gmsh.model.occ.addPoint(0.5, 0, -1, h, 2) # control point
    gmsh.model.occ.addPoint(1, 0, 0, h, 3)

    # curve
    gmsh.model.occ.addBezier([1, 2, 3], 1)

    # revolve curve around z-axis (makes surface 1)
    gmsh.model.occ.revolve([(1, 1)], 0, 0, -1, 0, 0, 1, 2π)

    # define curve loop made by revolving point 3 around z-axis as curve 2 
    gmsh.model.occ.addCurveLoop([3], 2)

    # fill in the circle made by curve 2 (surface 2)
    gmsh.model.occ.addSurfaceFilling(2, 2)

    # surfaces 1 and 2 make up the bowl
    gmsh.model.occ.addSurfaceLoop(1:2, 1)
    gmsh.model.occ.addVolume([1], 1)

    # sync and define physical groups
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(0, [1, 3], 2, "bot")
    gmsh.model.addPhysicalGroup(1, [1, 3], 2, "bot")
    gmsh.model.addPhysicalGroup(2, [2], 1, "sfc")
    gmsh.model.addPhysicalGroup(2, [3], 2, "bot")
    gmsh.model.addPhysicalGroup(3, [1], 3, "int")

    # generate and save mesh
    gmsh.model.mesh.generate(3)
    gmsh.write("bowl3D.msh")
    gmsh.finalize()
end

generate_bowl_mesh(0.05)