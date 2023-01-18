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

    # normals = [sign(cross(vcat(p_circ[t_circ[k, 1], :] - p_circ[t_circ[k, 2], :], 0),
    #                       vcat(p_circ[t_circ[k, 1], :] - p_circ[t_circ[k, 3], :], 0))[3]) for k ∈ axes(t_circ, 1)]
    # println(all(normals .== 1)) # returns true → all triangles have counterclockwise orientation

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
        # # local delaunay tessellation of column
        # i_col = vcat(tri_to_p[k, 1][:], tri_to_p[k, 2][:], tri_to_p[k, 3][:])
        # t_col = delaunay(p[i_col, :]).simplices

        # # remove zeros vols
        # t_col_g = i_col[t_col]
        # vols = [abs(det([p[t_col_g[k, j+1], i] - p[t_col_g[k, 1], i] for i=1:3, j=1:3])) for k ∈ axes(t_col_g, 1)]
        # keep = vols .!= 0
        # if !all(keep)
        #     # save vtu of bad egg
        #     cells = [MeshCell(VTKCellTypes.VTK_TETRA, t_col_g[i, :]) for i ∈ axes(t_col_g, 1)]
        #     vtk_grid("col$k.vtu", p', cells) do vtk
        #         bad_egg = zeros(size(p, 1))
        #         bad_egg[t_col_g[.!keep, :]] .= 1
        #         vtk["bad egg"] = bad_egg
        #     end
        #     println("col$k.vtu")

        #     # just tessellate first prism?
        #     i_col_12 = vcat(tri_to_p[k, 1][1:2], tri_to_p[k, 2][1:2], tri_to_p[k, 3][1:2])
        #     t_col_12 = delaunay(p[i_col_12, :]).simplices
        #     t_col_12g = i_col_12[t_col_12]
        #     cells = [MeshCell(VTKCellTypes.VTK_TETRA, t_col_12g[i, :]) for i ∈ axes(t_col_12g, 1)]
        #     vtk_grid("col$(k)_top.vtu", p', cells) do vtk
        #     end
        #     println("col$(k)_top.vtu")

        #     # mesh faces
        #     lens = length.(tri_to_p[k, :])
        #     tri_edges = [(1,2), (2,3), (3,1)]
        #     faces = []
        #     ts = []
        #     for (i, (e1, e2)) in enumerate(tri_edges)
        #         face = vcat(p[tri_to_p[k, e1][:], :], p[tri_to_p[k, e2][:], :])
        #         l = norm(face[1, :] - face[lens[i]+1, :])
        #         face[1:lens[i], 1] .= isempty(faces) ? 0 : faces[end][end, 1]
        #         face[lens[i]+1:end, 1] .= l + face[1, 1]
        #         face = face[:, [1, 3]]
        #         push!(faces, face)
        #         push!(ts, delaunay(face).simplices)
        #     end

        #     # plot
        #     fig, ax = subplots(1)
        #     for i=1:3
        #         tplot(faces[i], ts[i], fig=fig, ax=ax)
        #         ax.plot(faces[i][:, 1], faces[i][:, 2], "o", ms=1)
        #     end
        #     ax.set_xlim(-0.1, faces[3][end, 1]+0.1)
        #     ax.set_ylim(-1.1, 0.1)
        #     savefig("faces$k.png")
        #     plt.close()
        # end
        # t_col = t_col[keep, :]

        # # add to global t
        # t = [t; i_col[t_col]]
        # column lengths
        lens = length.(tri_to_p[k, :])

        # first top tri is at sfc
        top = [tri_to_p[k, i][1] for i=1:3]

        # continue down to bottom
        for j=2:maximum(lens)
            # make bottom tri from next nodes down or top tri nodes
            bot = [j ≤ lens[i] ? tri_to_p[k, i][j] : top[i] for i=1:3]

            # use delaunay to tessellate
            ig = unique(vcat(top, bot))
            tl = delaunay(p[ig, :]).simplices

            # add to t
            t = [t; ig[tl]]

            # continue
            top = bot
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

# p, t, e = valign3D("circle/mesh2.h5"; savefile="mesh.h5")

# fmap, faces, bndix = all_faces(t)
# cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, faces[i, :]) for i in bndix]
# vtk_grid("faces.vtu", p', cells) do vtk
# end
# println("faces.vtu")

# cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh1.vtu", p', cells) do vtk
# end
# println("mesh1.vtu")

# p, t, e = nuPGCM.add_nodes(p, t, e, 2)
# cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh2.vtu", p', cells) do vtk
#     bdy = zeros(size(p, 1))
#     bdy[e] .= 1
#     vtk["boundary"] = bdy
# end
# println("mesh2.vtu")

for i=0:2
    valign3D("circle/mesh$i.h5"; savefile="valign3D/mesh$i.h5")
end


println("Done.")