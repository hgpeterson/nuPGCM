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
    gmsh.write("meshes/mesh.msh")
    gmsh.finalize()
end

# function load_msh(ifile)
#     # number of dimensions
#     dim = 2

#     # initialize mesh and load from file
#     gmsh.initialize()
#     gmsh.option.setNumber("General.Terminal", 1)
#     gmsh.model.add("quad_circ_mesh")
#     gmsh.open(ifile)

#     # find node positions by looping through indices
#     np = gmsh.model.mesh.get_max_node_tag()
#     p = zeros(np, dim)
#     for i=1:np
#         coord, parametricCoord, dim, tag = gmsh.model.mesh.getNode(i)
#         p[i, :] = coord
#     end

#     # get tri
#     elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)
#     nodes = nodeTags[1]
#     nt = Int64(size(nodes, 1)/(dim + 1))
#     t = Array(Int64.(reshape(nodes, (dim+1, nt)))')

#     # edge nodes
#     nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodesByElementType(1)
#     e = unique!(Int64.(nodeTags))
#     println(gmsh.model.getPhysicalGroups(0))

#     gmsh.finalize()

#     return p, t, e
# end

# function msh2h5(ifile, ofile)
#     p, t, e = load_msh(ifile)
#     file = h5open(ofile, "w")
#     write(file, "p", p)
#     write(file, "t", t)
#     write(file, "e", e)
#     close(file)
#     println(ofile)
# end

# function msh2vtu(ifile, ofile)
#     # load p, t, e
#     p, t, e = load_msh(ifile)
#     np = size(p, 1)

#     # define points and cells for vtk
#     points = p'
#     cells = Vector{MeshCell}([])
#     for i in axes(t, 1)
#         push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]))
#     end

#     # save as vtu file
#     vtk_grid(ofile, points, cells) do vtk
#         boundary = zeros(np)
#         boundary[e] .= 1
#         vtk["boundary"] = boundary
#     end
# end

generate_circle_mesh(0.04)