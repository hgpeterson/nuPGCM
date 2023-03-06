using nuPGCM
using HDF5
using PyPlot
using Printf
using ProgressMeter

include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_col(h, H; order)
    # surface triangle 
    p_sfc = [0      0
             2h/√3  0
             h/√3   h]

    # depths
    Hs = H.(p_sfc[:, 1], p_sfc[:, 2])

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

function main()
    # Ekman
    ε² = 1

    # depth
    H(x, y) = 1 - x^2 - y^2

    # grid
    col, stacks = gen_col(0.1, H, order=1)
    nzs = [size(s, 1) for s ∈ stacks]

    # buoyancy
    x = col.p[:, 1]
    y = col.p[:, 2]
    z = col.p[:, 3]
    b = FEField(exp.(z).*sin.(x).*cos.(y), col)

    # buoyancy gradients
    # bx = []
    # by = []
    # for i ∈ eachindex(stacks)
    #     push!(bx, [∂x(b, (stacks[i][k, :] + stacks[i][k+1, :])/2) for k=1:nzs[i]-1])
    #     push!(by, [∂y(b, (stacks[i][k, :] + stacks[i][k+1, :])/2) for k=1:nzs[i]-1])
    # end
    bx = Vector{Vector{Float64}}(undef, size(stacks, 1))
    by = Vector{Vector{Float64}}(undef, size(stacks, 1))
    n = 0
    for i ∈ axes(stacks, 1)
        # bx[i] = zeros(nzs[i]-1)
        # by[i] = zeros(nzs[i]-1)
        # for j=1:nzs[i]-1
        #     k_tet = findfirst(k_tet -> n+j ∈ col.t[k_tet, :] && n+j+1 ∈ col.t[k_tet, :], 1:col.nt)
        #     bx[i][j] = ∂x(b, [0, 0, 0], k_tet)
        #     by[i][j] = ∂y(b, [0, 0, 0], k_tet)
        # end
        # n += nzs[i]

        x = stacks[i][1, 1]
        y = stacks[i][1, 2]
        z_half = (stacks[i][1:end-1, 3] + stacks[i][2:end, 3])/2
        bx[i] = @.  exp(z_half)*cos(x)*cos(y)
        by[i] = @. -exp(z_half)*sin(x)*sin(y)
    end

    # transports
    Ux = 0
    Uy = 0

    # wind stress
    τx = 0
    τy = 0

    # solve
    sols = [nzs[i] == 1 ? [0.0, 0.0] : solve_baroclinic_1dfe(stacks[i][:, 3], bx[i], by[i], Ux, Uy, τx, τy, ε²) for i ∈ eachindex(stacks)]

    # unpack
    ωxs = Vector{FEField}(undef, 3)
    ωys = Vector{FEField}(undef, 3)
    for i=1:3
        nz = nzs[i]
        p = reshape(stacks[i][:, 3], (nz, 1))
        t = [i + j - 1 for i=1:nz-1, j=1:2]
        e = Dict("bot"=>[nz], "sfc"=>[1])
        g = FEGrid(1, p, t, e)
        ωxs[i] = FEField(sols[i][1:nzs[i]], g)
        ωys[i] = FEField(sols[i][nzs[i]+1:end], g)
    end

    # plot column
    cell_type = col.order == 1 ? VTKCellTypes.VTK_TETRA : VTKCellTypes.VTK_QUADRATIC_TETRA
    cells = [MeshCell(cell_type, col.t[i, :]) for i ∈ axes(col.t, 1)]
    vtk_grid("output/pg_vort_DG_3D.vtu", col.p', cells) do vtk
        vtk["ωx"] = [ωxs[1].values; ωxs[2].values; ωxs[3].values]
        vtk["ωy"] = [ωys[1].values; ωys[2].values; ωys[3].values]
    end
    println("output/pg_vort_DG_3D.vtu")

    for i=1:3
        # fd sol 
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
        savefig("scratch/images/bj$i.png")
        println("scratch/images/bj$i.png")
        plt.close()

        # plot sol
        fig, ax = subplots(1, figsize=(2.2, 3))
        ax.plot(ωxs[i].values, ωxs[i].g.p, label=L"\omega^x")
        ax.plot(ωys[i].values, ωxs[i].g.p, label=L"\omega^y")
        ax.plot(ωx_fd, z, "k--", lw=0.5, label="FD Sol")
        ax.plot(ωy_fd, z, "k--", lw=0.5)
        ax.set_xlabel(L"\omega")
        ax.set_ylabel(L"z")
        ax.legend()
        savefig("scratch/images/omega$i.png")
        println("scratch/images/omega$i.png")
        plt.close()

        # error
        println(@sprintf("Max error ωx: %1.1e", maximum(x->isnan(x) ? -Inf : x, abs.(ωxs[i].(z) - ωx_fd))))
        println(@sprintf("Max error ωy: %1.1e", maximum(x->isnan(x) ? -Inf : x, abs.(ωys[i].(z) - ωy_fd))))
    end
end

main()
println("Done.")