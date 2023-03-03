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
    Hs = [1, 1.1, 1]

    # number of nodes in vertical
    nzs = Int64.(ceil.(Hs./h))

    # stacks
    stacks = Vector{Matrix{Float64}}(undef, 3)

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
        stacks[i] = [p_sfc[i, 1]*ones(nz)  p_sfc[i, 2]*ones(nz)  z]
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
col, stacks = gen_col(0.01, order=1)
nzs = [size(s, 1) for s ∈ stacks]

# buoyancy
x = col.p[:, 1]
y = col.p[:, 2]
z = col.p[:, 3]
b = FEField(exp.(z).*sin.(x).*cos.(y), col)

# buoyancy gradients
bx = []
by = []
for i ∈ eachindex(stacks)
    push!(bx, [∂x(b, (stacks[i][k, :] + stacks[i][k+1, :])/2) for k=1:nzs[i]-1])
    push!(by, [∂y(b, (stacks[i][k, :] + stacks[i][k+1, :])/2) for k=1:nzs[i]-1])
end

# transports
Ux = 0
Uy = 0

# solve
sols = [nzs[i] == 1 ? [0.0, 0.0] : solve_baroclinic_1dfe(stacks[i][:, 3], bx[i], by[i], Ux, Uy, ε²) for i ∈ eachindex(stacks)]

# plot column
cell_type = col.order == 1 ? VTKCellTypes.VTK_TETRA : VTKCellTypes.VTK_QUADRATIC_TETRA
cells = [MeshCell(cell_type, col.t[i, :]) for i ∈ axes(col.t, 1)]
vtk_grid("output/pg_vort_DG_3D.vtu", col.p', cells) do vtk
    vtk["ωx"] = [sols[1][1:nzs[1]]; sols[2][1:nzs[2]]; sols[3][1:nzs[3]]]
    vtk["ωy"] = [sols[1][nzs[1]+1:end]; sols[2][nzs[2]+1:end]; sols[3][nzs[3]+1:end]]
end
println("output/pg_vort_DG_3D.vtu")

# fd sol 
i = 1
x = stacks[i][1, 1]
y = stacks[i][1, 2]
z = range(stacks[i][end, 3], stacks[i][1, 3], length=2^10)
bx_a = @. exp(z)*cos(x)*cos(y)
by_a = @. -exp(z)*sin(x)*sin(y)
ωx_fd, ωy_fd = solve_baroclinic_1dfd(z, bx_a, by_a, Ux, Uy, ε²)

# plot b grads
fig, ax = subplots(1, figsize=(2.2, 3))
z_half = (stacks[i][1:end-1, 3] + stacks[i][2:end, 3])/2
ax.plot(bx[i], z_half, "o", ms=1, label=L"\partial_x b")
ax.plot(by[i], z_half, "o", ms=1, label=L"\partial_y b")
ax.plot(bx_a, z, "tab:blue", ls="--", lw=0.5, label="Truth")
ax.plot(by_a, z, "tab:orange", ls="--", lw=0.5)
ax.set_xlabel(L"\partial_j b")
ax.set_ylabel(L"z")
ax.legend()
savefig("scratch/images/bj.png")
println("scratch/images/bj.png")
plt.close()

# plot sol
fig, ax = subplots(1, figsize=(2.2, 3))
ax.plot(sols[i][1:nzs[i]], stacks[i][:, 3], label=L"\omega^x")
ax.plot(sols[i][nzs[i]+1:end], stacks[i][:, 3], label=L"\omega^y")
ax.plot(ωx_fd, z, "k--", lw=0.5, label="FD Sol")
ax.plot(ωy_fd, z, "k--", lw=0.5)
ax.set_xlabel(L"\omega")
ax.set_ylabel(L"z")
ax.legend()
savefig("scratch/images/omega.png")
println("scratch/images/omega.png")
plt.close()

println("Done.")