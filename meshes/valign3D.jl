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
    println(h)

    # depth
    H = @. 1 - x^2 - y^2

    # first add coastline
    p = [x[e_circ]  y[e_circ]  zeros(ne_circ)]
    e = collect(1:ne_circ)

    # then add interior points
    for i=interior
        n = size(p, 1)
        nz = Int64(ceil(H[i]/h))
        # if nz == 2
        #     nz += 1
        # end
        z = range(-H[i], 0, length=nz)
        # z = @. -H[i]*(cos(π*(0:nz-1)/(nz-1)) + 1)/2  
        p = vcat(p, [x[i]*ones(nz)  y[i]*ones(nz)  z])
        push!(e, n+1)
        push!(e, size(p, 1))
    end

    # delaunay tesselation (for now)
    mesh = delaunay(p)
    t = mesh.simplices

    # # remove bad tetras
    # nt = size(t, 1)
    # mask = ones(Bool, nt)
    # for k=1:nt
    #     A = [p[t[k, i+1], j] - p[t[k, 1], j] for i=1:3, j=1:3]
    #     # if det(A) == 0 # volume of tet = 0
    #     if abs(det(A)) ≤ 1e-4 # volume of tet = 0
    #         println("Removing tet $k")
    #         mask[k] = 0
    #     end
    # end
    # t = t[mask, :]

    println("np = ", size(p, 1))

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
for i=0:5
    p, t, e = valign3D("circle/mesh$i.h5"; savefile="valign3D/mesh$i.h5")
end

# cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh1.vtu", p', cells) do vtk
#     bdy = zeros(size(p, 1))
#     bdy[e] .= 1
#     vtk["boundary"] = bdy
# end

# p, t, e = nuPGCM.add_nodes(p, t, e, 2)
# cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, t[i, :]) for i in axes(t, 1)]
# vtk_grid("mesh2.vtu", p', cells) do vtk
#     bdy = zeros(size(p, 1))
#     bdy[e] .= 1
#     vtk["boundary"] = bdy
# end

println("Done.")