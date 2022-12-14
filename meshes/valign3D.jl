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
    p_circ = Matrix{Float64}[]
    t_circ = Matrix{Int64}[]
    e_circ = Vector{Int64}[]
    h5open(ifile, "r") do file
        p_circ = read(file, "p")
        t_circ = Int64.(read(file, "t"))
        e_circ = Int64.(read(file, "e")[:, 1])
    end
    x = p_circ[:, 1]
    y = p_circ[:, 2]
    np_circ = size(p_circ, 1)
    ne_circ = size(e_circ, 1)

    # interior
    interior = findall(!in(e_circ), 1:np_circ)

    # mesh res
    emap, edges, bndix = all_edges(t_circ)
    h = sum(norm(p_circ[edges[i, 1], :] - p_circ[edges[i, 2], :]) for i in axes(edges, 1))/size(edges, 1)

    # depth
    H = @. 1 - x^2 - y^2

    # mapping from points to triangles:
    # `p_to_tri[i]` = vector of cartesian indices pointing to where point `i` is in `t_circ`
    p_to_tri = [findall(I -> i ∈ t_circ[I], CartesianIndices(size(t_circ))) for i ∈ axes(p_circ, 1)]

    # mapping from triangles to points in 3D: 
    # `tri_to_p[k, i][j]` = the `j`th point in the vertical for the `i`th point of triangle `k`
    # begin by populating it with the surface nodes `t_circ`
    tri_to_p = [[t_circ[k, i]] for k ∈ axes(t_circ, 1), i ∈ axes(t_circ, 2)]

    # add coastline to p and e
    p = [x[e_circ]  y[e_circ]  zeros(ne_circ)]
    e = collect(1:ne_circ)

    # add interior points to p, e, and tri_to_p
    for i=interior
        # current index
        np = size(p, 1)

        # vertical grid
        nz = Int64(ceil(H[i]/h))
        z = range(-H[i], 0, length=nz)

        # add to p
        p = vcat(p, [x[i]*ones(nz)  y[i]*ones(nz)  z])

        # add to e
        push!(e, np + 1)
        push!(e, np + nz)

        # add to tri_to_p
        for I ∈ p_to_tri[i]
            for j=np+2:np+nz
                push!(tri_to_p[I], j)
            end
        end
    end
    println("np = ", size(p, 1))

    # # delaunay tesselation (for now)
    # mesh = delaunay(p)
    # t = mesh.simplices

    # compute tesselation
    # for k ∈ axes(tri_to_p, 1)
    #     println([length(tri_to_p[k, i]) for i=1:3])
    # end
    t = Matrix{Int64}[]
    for k ∈ axes(tri_to_p, 1)
        if length(tri_to_p[k, 1]) == length(tri_to_p[k, 2]) == length(tri_to_p[k, 3])
            for j=1:length(tri_to_p[k, 1])-1
                if isempty(t)
                    t = [tri_to_p[k, 1][j] tri_to_p[k, 2][j]   tri_to_p[k, 3][j]   tri_to_p[k, 1][j+1]]
                else
                    t = [t; tri_to_p[k, 1][j] tri_to_p[k, 2][j]   tri_to_p[k, 3][j]   tri_to_p[k, 1][j+1]]
                end
                t = [t; tri_to_p[k, 2][j] tri_to_p[k, 3][j]   tri_to_p[k, 1][j+1] tri_to_p[k, 2][j+1]]
                t = [t; tri_to_p[k, 3][j] tri_to_p[k, 1][j+1] tri_to_p[k, 2][j+1] tri_to_p[k, 3][j+1]]
            end
        end
    end

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

p, t, e = valign3D("circle/mesh1.h5"; savefile="mesh.h5")

# for i=0:5
#     p, t, e = valign3D("circle/mesh$i.h5"; savefile="valign3D/mesh$i.h5")
# end

cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i in axes(t, 1)]
vtk_grid("mesh1.vtu", p', cells) do vtk
end

# p, t, e = nuPGCM.add_nodes(p, t, e, 2)
# cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh2.vtu", p', cells) do vtk
#     bdy = zeros(size(p, 1))
#     bdy[e] .= 1
#     vtk["boundary"] = bdy
# end

println("Done.")