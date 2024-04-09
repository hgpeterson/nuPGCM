using nuPGCM
using PyPlot
using HDF5
# import Gmsh: gmsh

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false) 

function load_msh(ifile)
    # initialize mesh and load from file
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("mesh")
    gmsh.open(ifile)

    # find node positions by looping through indices
    np = gmsh.model.mesh.get_max_node_tag()
    p = zeros(np, 3)
    for i=1:np
        coord, parametricCoord, dim, tag = gmsh.model.mesh.getNode(i)
        p[i, :] = coord[1:3]
    end

    # get tri
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2)
    nodes = nodeTags[1]
    nt = Int64(size(nodes, 1)/3)
    t = Array(Int64.(reshape(nodes, (3, nt)))')

    # edge nodes
    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodesByElementType(1)
    e = unique!(Int64.(nodeTags))
    println(gmsh.model.getPhysicalGroups(0))

    gmsh.finalize()

    return p, t, e
end

function msh2h5(ifile, ofile)
    p, t, e = load_msh(ifile)
    file = h5open(ofile, "w")
    write(file, "p", p)
    write(file, "t", t)
    write(file, "e", e)
    close(file)
    println(ofile)
end

function plot_mesh(p, t, e; fname="mesh.png")
    fig, ax, img = nuPGCM.tplot(p, t)
    ax.plot(p[e, 1], p[e, 2], ".", ms=1)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    savefig(fname)
    println(fname)
end

function make_meshes(mesh_maker::Function, shape)
    for i=0:5
        # res
        # h = 1e-2*2^(5-i)
        h = 1e-2*2. ^(3-i)

        # make mesh.msh file
        mesh_maker(h)

        # debug plot
        p, t, e = load_msh("mesh.msh")
        plot_mesh(p, t, e; fname="$(string(shape))/mesh$i.png")

        # save
        msh2h5("mesh.msh", "$(string(shape))/mesh$i.h5")
    end
end