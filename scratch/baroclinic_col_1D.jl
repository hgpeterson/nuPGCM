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
    # Hs = [1, 1, 1]
    Hs = [0, 0.2, 0.2]

    # number of nodes in vertical
    # nzs = Int64.(ceil.(Hs./h))
    nzs = [1, 3, 3]

    # stacks
    stacks = Vector{Vector{Float64}}(undef, 3)

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

        # save stack
        stacks[i] = collect(z)
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

    return col, stacks
end

# Ekman
ε² = 1

# grid
col, stacks = gen_col(0.1, order=1)
nzs = [size(s, 1) for s ∈ stacks]

# buoyancy
b = zeros(col.np)

# transports
Ux = ones(col.np)
Uy = zeros(col.np)

# solve
sols = [nzs[i] == 1 ? [0.0, 0.0] : solve_baroclinic_1dfe(stacks[i], zeros(nzs[i]), zeros(nzs[i]), 1, 0, ε²) for i ∈ eachindex(stacks)]

# plot
cell_type = col.order == 1 ? VTKCellTypes.VTK_TETRA : VTKCellTypes.VTK_QUADRATIC_TETRA
cells = [MeshCell(cell_type, col.t[i, :]) for i ∈ axes(col.t, 1)]
vtk_grid("output/pg_vort_DG_3D.vtu", col.p', cells) do vtk
    vtk["ωx"] = [sols[1][1:nzs[1]]; sols[2][1:nzs[2]]; sols[3][1:nzs[3]]]
    vtk["ωy"] = [sols[1][nzs[1]+1:end]; sols[2][nzs[2]+1:end]; sols[3][nzs[3]+1:end]]
end
println("output/pg_vort_DG_3D.vtu")

# fig, ax = subplots(1, figsize=(2.2, 3))
# ax.plot(ωx.values, g.p[:, 1], label=L"\omega^x")
# ax.plot(ωy.values, g.p[:, 1], label=L"\omega^y")
# ax.set_xlabel(L"\omega")
# ax.set_ylabel(L"z")
# ax.legend()
# savefig("scratch/images/omega.png")
# println("scratch/images/omega.png")
# plt.close()

println("Done.")