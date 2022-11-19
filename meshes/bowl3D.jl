using nuPGCM
using HDF5
using PyPlot
using WriteVTK
import Gmsh: gmsh

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

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

    # bottom
    gmsh.model.addPhysicalGroup(0, 1:2, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, 1:2, 1, "bottom")
    gmsh.model.addPhysicalGroup(2, [1], 1, "bottom")
    # surface
    gmsh.model.addPhysicalGroup(2, [2], 2, "surface")
    # interior
    gmsh.model.addPhysicalGroup(3, [3], 3, "interior")

    # mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h₀)

    # generate mesh
    gmsh.model.mesh.generate(3)
    
    # save
    gmsh.write("mesh.msh")
    gmsh.finalize()
end

function load_msh(ifile)
    # number of dimensions
    dim = 3

    # initialize mesh and load from file
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("bowl_mesh")
    gmsh.open(ifile)

    # find node positions by looping through indices
    np = gmsh.model.mesh.get_max_node_tag()
    p = zeros(np, dim)
    for i=1:np
        p[i, :], param_coord, dim, tag = gmsh.model.mesh.getNode(i)
    end

    # get tetrahedra
    for i=0:3
        el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(i)
        println(el_types)
    end
    el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(3)
    tri_nodes = el_node_tags[1]
    nt = Int64(size(tri_nodes, 1)/(dim + 1))
    t = Array(Int64.(reshape(tri_nodes, (dim+1, nt)))')

    # edge nodes
    e, e_coords = gmsh.model.mesh.getNodesByElementType(2)
    ebot, ebot_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)
    etop, etop_coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, 2)
    e = Int64.(e)
    ebot = Int64.(ebot)
    etop = Int64.(etop)
    unique!(e)
    unique!(ebot)
    unique!(etop)

    gmsh.finalize()

    return p, t, e, ebot, etop
end

function msh2h5(ifile, ofile)
    p, t, e, ebot, etop = load_msh(ifile)
    file = h5open(ofile, "w")
    write(file, "p", p)
    write(file, "t", t)
    write(file, "e", e)
    close(file)
    println(ofile)
end

function msh2vtu(ifile, ofile)
    # load p, t, e
    p, t, e, ebot, etop = load_msh(ifile)
    np = size(p, 1)

    # define points and cells for vtk
    points = p'
    cells = Vector{MeshCell}([])
    for i in axes(t, 1)
        push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]))
    end

    # save as vtu file
    vtk_grid(ofile, points, cells) do vtk
        bot = zeros(np)
        bot[ebot] .= 1
        vtk["bot"] = bot
        top = zeros(np)
        top[etop] .= 1
        vtk["top"] = top
        boundary = zeros(np)
        boundary[e] .= 1
        vtk["boundary"] = boundary
    end
end

# hs = [0.16, 0.08, 0.04, 0.02, 0.01]
# for i in eachindex(hs)
#     generate_bowl_mesh(hs[i])
#     msh2h5("mesh.msh", "bowl3D/mesh$(i-1).h5")
# end

generate_bowl_mesh(0.16)

# p, t, e, ebot, etop = load_msh("mesh.msh")

# msh2h5("mesh.msh", "mesh.h5")

msh2vtu("mesh.msh", "mesh.vtu")