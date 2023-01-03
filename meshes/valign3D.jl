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
    p_to_tri = [findall(I -> i ∈ t_circ[I], CartesianIndices(size(t_circ))) for i ∈ axes(p_circ, 1)]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    # begin by populating it with the surface nodes `t_circ`
    tri_to_p = [[t_circ[k, i]] for k=1:nt_circ, i=1:3]

    # add coastline to p and e
    p = [x[e_circ]  y[e_circ]  zeros(ne_circ)]
    e = collect(1:ne_circ)

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

    # # delaunay tesselation (for now)
    # mesh = delaunay(p)
    # t = mesh.simplices

    # compute tesselation
    t = [0 0 0 0] # allocate
    for k=1:nt_circ
        # column lengths
        lens = length.(tri_to_p[k, :])

        # first top tri is at sfc (j = 1)
        top = [tri_to_p[k, i][1] for i=1:3]

        # continue down to bottom
        for j=2:maximum(lens)
            # make bottom tri from next nodes down or top tri nodes
            bot = [j ≤ lens[i] ? tri_to_p[k, i][j] : top[i] for i=1:3]

            # add to t
            t = [t; tessellate(top, bot)]

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

function tessellate(top, bot)
    # get unique bot indices
    bot = [i for i ∈ bot if i ∉ top]
    if length(bot) == 3
        return [top[1] top[2] top[3] bot[1]
                top[2] top[3] bot[1] bot[2]
                top[3] bot[1] bot[2] bot[3]]
    elseif length(bot) == 2
        return [top[1] top[2] top[3] bot[1]
                top[2] top[3] bot[1] bot[2]]
    elseif length(bot) == 1
        return [top[1] top[2] top[3] bot[1]]
    end
end

p, t, e = valign3D("circle/mesh1.h5"; savefile="mesh.h5")

# for i=0:5
#     p, t, e = valign3D("circle/mesh$i.h5"; savefile="valign3D/mesh$i.h5")
# end

cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i in axes(t, 1)]
vtk_grid("mesh.vtu", p', cells) do vtk
end
println("mesh.vtu")

# p, t, e = nuPGCM.add_nodes(p, t, e, 2)
# cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh2.vtu", p', cells) do vtk
#     bdy = zeros(size(p, 1))
#     bdy[e] .= 1
#     vtk["boundary"] = bdy
# end

println("Done.")