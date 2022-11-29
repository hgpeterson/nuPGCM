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
    gmsh.model.add("mesh")

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
    # physical group tags
    bottom = 1
    surface = 2
    interior = 3

    # initialize mesh and load from file
    gmsh.initialize()
    gmsh.open(ifile)

    # get all node positions (`pts[i, 1:3]` are the coordinates of point `i`)
    pts = Array(reshape(gmsh.model.mesh.getNodes()[2], (3, :))')

    # get interior tetrahedra
    tets = get_elements(3, interior)

    # get boundary triangles
    tris_bot = get_elements(2, bottom)
    tris_sfc = get_elements(2, surface)

    # boundary nodes
    bdy_bot = Int64.(gmsh.model.mesh.getNodesForPhysicalGroup(2, bottom)[1])
    bdy_sfc = Int64.(gmsh.model.mesh.getNodesForPhysicalGroup(2, surface)[1])

    # finalize
    gmsh.finalize()

    return pts, tets, tris_bot, tris_sfc, bdy_bot, bdy_sfc
end

function get_elements(dim, tag)
    # get nodes in elements of dimension `dim` with tag `tag`
    nodes = gmsh.model.mesh.getElements(dim, tag)[3][1]

    # `els[i, j]` is the index of node `j` in element `i`
    els = Array(Int64.(reshape(nodes, (dim+1, :)))')
    return els
end

function msh2h5(ifile, ofile)
    pts, tets, tris_bot, tris_sfc, bdy_bot, bdy_sfc = load_msh(ifile)
    file = h5open(ofile, "w")
    write(file, "pts", pts)
    write(file, "tets", tets)
    write(file, "tris_bot", tris_bot)
    write(file, "tris_sfc", tris_sfc)
    write(file, "bdy_bot", bdy_bot)
    write(file, "bdy_sfc", bdy_sfc)
    close(file)
    println(ofile)
end

function msh2vtu(ifile, ofile)
    # load .msh file
    pts, tets, tris_bot, tris_sfc, bdy_bot, bdy_sfc = load_msh(ifile)

    # vtk takes points as column vectors
    pts = pts'

    # define cells for vtk
    cells = [MeshCell(VTKCellTypes.VTK_TETRA, tets[i, :]) for i in axes(tets, 1)]

    # save as vtu file
    vtk_grid(ofile, pts, cells) do vtk
        sfc = zeros(size(pts, 2))
        sfc[bdy_sfc] .= 1
        vtk["sfc"] = sfc
        bot = zeros(size(pts, 2))
        bot[bdy_bot] .= 1
        vtk["bot"] = bot
    end
end

# hs = [0.16, 0.08, 0.04, 0.02, 0.01]
# for i in eachindex(hs)
#     generate_bowl_mesh(hs[i])
#     msh2h5("mesh.msh", "bowl3D/mesh$(i-1).h5")
# end

# generate_bowl_mesh(0.16)

# pts, tets, tris_bot, tris_sfc, bdy_bot, bdy_sfc = load_msh("mesh.msh")

msh2h5("mesh.msh", "mesh.h5")

# msh2vtu("mesh.msh", "mesh.vtu")