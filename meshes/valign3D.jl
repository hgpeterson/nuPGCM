using nuPGCM
using LinearAlgebra
using Delaunay
using HDF5
using PyPlot
using WriteVTK

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function valign3D(ifile; savefile=nothing)
    # load mesh of circle
    file = h5open(ifile, "r")
    p_circ = read(file, "p")
    t_circ = Int64.(read(file, "t"))
    e_circ = Int64.(read(file, "e")[:, 1])
    close(file)
    x = p_circ[:, 1]
    y = p_circ[:, 2]
    np_circ = size(p_circ, 1)
    nt_circ = size(t_circ, 1)
    ne_circ = size(e_circ, 1)

    # interior
    interior = findall(!in(e_circ), 1:np_circ)

    # mesh res
    emap, edges, bndix = all_edges(t_circ)
    h = sum(norm(p_circ[edges[i, 1], :] - p_circ[edges[i, 2], :]) for i in axes(edges, 1))/size(edges, 1)

    # depth
    H = @. 1 - x^2 - y^2

    # mapping from points to triangles:
    #   `p_to_tri[i]` is vector of cartesian indices pointing to where point `i` is in `t_circ`
    p_to_tri = [findall(I -> i ∈ t_circ[I], CartesianIndices(size(t_circ))) for i=1:np_circ]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    tri_to_p = [Int64[] for k=1:nt_circ, i=1:3] # allocate

    # add coastline to p and e
    p = [x[e_circ]  y[e_circ]  zeros(ne_circ)]
    e = collect(1:ne_circ)

    # add coastline to tri_to_p
    for j=1:ne_circ
        for I ∈ p_to_tri[e_circ[j]]
            push!(tri_to_p[I], j)
        end
    end

    # add interior points to p, e, and tri_to_p
    for i=interior
        # current index
        np = size(p, 1)

        # vertical grid
        nz = Int64(ceil(H[i]/h))
        z = -range(0, H[i], length=nz)

        # add to p
        p = vcat(p, [x[i]*ones(nz)  y[i]*ones(nz)  z])

        # add to e
        push!(e, np + 1)
        push!(e, np + nz)

        # add to tri_to_p
        for I ∈ p_to_tri[i]
            for j=np+1:np+nz
                push!(tri_to_p[I], j)
            end
        end
    end
    println("np = ", size(p, 1))

    # compute tessellation
    t = [0 0 0 0] # allocate
    for k=1:nt_circ
        # local delaunay tessellation of column
        i_col = vcat(tri_to_p[k, 1][:], tri_to_p[k, 2][:], tri_to_p[k, 3][:])
        t_col = delaunay(p[i_col, :]).simplices

        # add to global t
        t = [t; i_col[t_col]]

        # compute volume of each tet
        nt = size(t, 1)
        vols = [abs(det([p[t[k, j+1], i] - p[t[k, 1], i] for i=1:3, j=1:3])) for k=nt-size(t_col, 1)+1:nt]
        if 0 ∈ vols
            cells = [MeshCell(VTKCellTypes.VTK_TETRA, t_col[i, :]) for i in axes(t_col, 1)]
            vtk_grid("col$k.vtu", p[i_col, :]', cells) do vtk
            end
            println("col$k.vtu")
        end
    end
    t = t[2:end, :] # remove init 0's

    if savefile !== nothing
        h5open(savefile, "w") do file
            write(file, "p", p)
            write(file, "t", t)
            write(file, "e", e)
        end
        println(savefile)
    end

    return p, t, e
end

p, t, e = valign3D("circle/mesh2.h5"; savefile="mesh.h5")

# fmap, faces, bndix = nuPGCM.all_faces(t)
# # cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, faces[i, :]) for i in axes(faces, 1)]
# cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, faces[i, :]) for i in bndix]
# vtk_grid("faces.vtu", p', cells) do vtk
# end
# println("faces.vtu")

cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i in axes(t, 1)]
vtk_grid("mesh1.vtu", p', cells) do vtk
end
println("mesh1.vtu")

# p, t, e = nuPGCM.add_nodes(p, t, e, 2)
# cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh2.vtu", p', cells) do vtk
#     bdy = zeros(size(p, 1))
#     bdy[e] .= 1
#     vtk["boundary"] = bdy
# end
# println("mesh2.vtu")

# for i=0:2
#     valign3D("circle/mesh$i.h5"; savefile="valign3D/mesh$i.h5")
# end


println("Done.")