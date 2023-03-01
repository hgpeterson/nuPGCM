using nuPGCM
using HDF5
using PyPlot
using Printf
using ProgressMeter

include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_col(h; order)
    # surface triangle 
    p_sfc = [0      0
             2h/√3  0
             h/√3   h]

    # depths
    Hs = [1, 1, 1]

    # number of nodes in vertical
    # nzs = Int64.(ceil.(Hs./h))
    nzs = [1, 2, 2]

    # add points in vertical
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>[1, nzs[1]+1, nzs[1]+nzs[2]+1], "bot"=>[nzs[1], nzs[1]+nzs[2], nzs[1]+nzs[2]+nzs[3]])
    np = 0
    for i=1:3
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0.0]
        else
            z = -range(0, Hs[i], length=nz)
        end

        # add to p
        p[np+1:np+nz, :] = [p_sfc[i, 1]*ones(nz)  p_sfc[i, 2]*ones(nz)  z]
        np += nz
    end

    # tessellate
    t = Matrix{Int64}(undef, 0, 4) 
    top = [1, 1+nzs[1], 1+nzs[1]+nzs[2]]
    for j=2:maximum(nzs)
        # make bottom tri from next nodes down or top tri nodes
        bot = [j ≤ nzs[i] ? top[i] + 1 : top[i] for i=1:3]

        # use delaunay to tessellate
        ig = unique(vcat(top, bot))
        tl = delaunay(p[ig, :]).simplices

        # add to t
        t = [t; ig[tl]]

        # continue
        top = bot
    end

    # column
    col = FEGrid(order, p, t, e)

    # remove from bot if in sfc
    col.e["bot"] = col.e["bot"][findall(i -> col.e["bot"][i] ∉ col.e["sfc"], 1:size(col.e["bot"], 1))]

    return col
end

# Ekman
ε² = 1

# grid
col = gen_col(1, order=2)

# buoyancy
b = zeros(col.np)

# transports
Ux = ones(col.np)
Uy = zeros(col.np)

# solve
sol = solve_baroclinic(col, b, Ux, Uy, ε²)

# plot
plot_1D(col, sol, (x, y) -> 1, (x, y, z) -> 0, (x, y, z) -> 0, (x, y) -> 1, (x, y) -> 0)
cell_type = col.order == 1 ? VTKCellTypes.VTK_TETRA : VTKCellTypes.VTK_QUADRATIC_TETRA
cells = [MeshCell(cell_type, col.t[i, :]) for i ∈ axes(col.t, 1)]
vtk_grid("output/pg_vort_DG_3D.vtu", col.p', cells) do vtk
    vtk["ωx"] = sol[0*col.np+1:1*col.np]
    vtk["ωy"] = sol[1*col.np+1:2*col.np]
    vtk["χx"] = sol[2*col.np+1:3*col.np]
    vtk["χy"] = sol[3*col.np+1:4*col.np]

    bdy = zeros(col.np)
    bdy[col.e["sfc"]] .= 1
    vtk["sfc"] = bdy

    bdy = zeros(col.np)
    bdy[col.e["bot"]] .= 1
    vtk["bot"] = bdy
end
println("output/pg_vort_DG_3D.vtu")

println("Done.")