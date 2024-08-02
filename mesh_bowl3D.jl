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
    # gmsh.initialize()
    # gmsh.model.add("bowl3D")
    # gmsh.model.geo.addPoint(-1, 0, 0, h, 1)
    # gmsh.model.geo.addPoint(0, 0, -2, h, 2)
    # gmsh.model.geo.addPoint(1, 0, 0, h, 3)
    # gmsh.model.geo.addBezier([1, 2, 3], 1)
    # gmsh.model.geo.addLine(3, 1, 2)
    # gmsh.model.geo.synchronize()
    # gmsh.model.addPhysicalGroup(1, [1, 2], 1)
    # ov1 = gmsh.model.geo.revolve([(1, 1), (1, 2)], 0, 0, -1, 0, 0, 1, π/2)
    # sfcs1 = findall(i -> ov1[i][1] == 2, 1:length(ov1))
    # ov2 = gmsh.model.geo.revolve(ov1[sfcs1 .- 1], 0, 0, -1, 0, 0, 1, π/2)
    # sfcs2 = findall(i -> ov2[i][1] == 2, 1:length(ov2))
    # gmsh.model.geo.synchronize()
    # sfc_tags = vcat([ov1[i][2] for i ∈ sfcs1], [ov2[i][2] for i ∈ sfcs2])
    # gmsh.model.addPhysicalGroup(2, sfc_tags, 1)
    # gmsh.model.geo.addSurfaceLoop(sfc_tags, 1)
    # gmsh.model.geo.addVolume([1], 1)
    # gmsh.model.geo.synchronize()
    # gmsh.model.addPhysicalGroup(3, [1], 1)
    # gmsh.model.mesh.generate(3)
    # gmsh.write("bowl3D.msh")
    # gmsh.finalize()

    # # init
    # gmsh.initialize()

    # # model
    # gmsh.model.add("bowl3D")

    # # points
    # gmsh.model.occ.addPoint(0, 0, -1, h, 1)
    # gmsh.model.occ.addPoint(0.5, 0, -1, h, 2) # control point
    # gmsh.model.occ.addPoint(1, 0, 0, h, 3)
    # gmsh.model.occ.addPoint(0, 0, 0, h, 4)

    # # curves
    # gmsh.model.occ.addBezier([1, 2, 3], 1)
    # gmsh.model.occ.addLine(3, 4, 2)

    # # revolve curves 1 and 2 about the z-axis
    # # ov = gmsh.model.occ.revolve([(1, 1), (1, 2)], 0, 0, -1, 0, 0, 1, 2π)
    # ov = gmsh.model.occ.revolve([(1, 1), (1, 2)], 0, 0, -1, 0, 0, 1, π)
    # display(ov)

    # gmsh.model.occ.addCurveLoop([1, 2, 5, 6], 3)
    # gmsh.model.occ.addSurfaceFilling(3, 3)

    # # combined 2D surfaces define boundary of 3D volume
    # # gmsh.model.occ.addSurfaceLoop([1, 2], 1)
    # gmsh.model.occ.addSurfaceLoop(1:3, 1)
    # gmsh.model.occ.addVolume([1], 1)

    # # label physical groups
    # gmsh.model.occ.synchronize()
    # gmsh.model.addPhysicalGroup(0, [1, 3], 1, "bot")
    # gmsh.model.addPhysicalGroup(0, [4], 2, "sfc")
    # # gmsh.model.addPhysicalGroup(1, [1, 4], 1, "bot")
    # # gmsh.model.addPhysicalGroup(1, [2], 2, "sfc")
    # gmsh.model.addPhysicalGroup(1, [1, 4, 5], 1, "bot")
    # gmsh.model.addPhysicalGroup(1, [2, 6], 2, "sfc")
    # # gmsh.model.addPhysicalGroup(2, [1, 3], 1, "bot")
    # gmsh.model.addPhysicalGroup(2, [1, 3, 4], 1, "bot")
    # gmsh.model.addPhysicalGroup(2, [2], 2, "sfc")
    # gmsh.model.addPhysicalGroup(3, [1], 3, "int")

    # for i ∈ 0:3
    #     println(gmsh.model.getEntities(i))
    # end

    # # generate mesh
    # gmsh.model.mesh.generate(3)

    # # save
    # gmsh.write("bowl3D.msh")
    # gmsh.finalize()

    gmsh.initialize()
    gmsh.model.add("bowl3D")
    gmsh.model.occ.addPoint(0, 0, -1, h, 1)
    gmsh.model.occ.addPoint(1, 0, 0, h, 2)
    gmsh.model.occ.addLine(1, 2, 1)
    ov = gmsh.model.occ.revolve([(1, 1)], 0, 0, -1, 0, 0, 1, 2π)
    gmsh.model.occ.addCurveLoop([3], 2)
    gmsh.model.occ.addSurfaceFilling(2, 2)
    gmsh.model.occ.addSurfaceLoop([1, 2], 1)
    gmsh.model.occ.addVolume([1], 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(0, 1:2, 1)
    gmsh.model.addPhysicalGroup(1, 1:3, 1)
    gmsh.model.addPhysicalGroup(2, 1:3, 1)
    gmsh.model.addPhysicalGroup(3, [1], 1)
    for i ∈ 0:3
        println(gmsh.model.getEntities(i))
    end
    gmsh.model.mesh.generate(3)
    _, p, _ = gmsh.model.mesh.getNodes()
    p = reshapoe(p, (:, 3))
    display(size(p))
    _, _, t = gmsh.model.mesh.getElements(3)
    display(size(t))
    gmsh.write("bowl3D.msh")
    gmsh.finalize()
end

generate_bowl_mesh(0.1)