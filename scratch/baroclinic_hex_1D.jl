using nuPGCM
using HDF5
using PyPlot
using Printf
using ProgressMeter

include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_hex(h, H; order)
    # surface triangle 
    # p_sfc = [  0     0
    #          -h/√3  -h
    #           h/√3  -h
    #          2h/√3   0
    #           h/√3   h
    #          -h/√3   h
    #         -2h/√3   0]
    # θs = cumsum((π/3 - π/5)*rand(5) .+ π/5)
    θs = cumsum([π/3, π/4, π/2, π/3, π/4])
    r = 2h/√3
    p_sfc = [0            0
             r            0
             r*cos(θs[1]) r*sin(θs[1])
             r*cos(θs[2]) r*sin(θs[2])
             r*cos(θs[3]) r*sin(θs[3])
             r*cos(θs[4]) r*sin(θs[4])
             r*cos(θs[5]) r*sin(θs[5])]
    t_sfc = [1 2 3
             1 3 4
             1 4 5
             1 5 6
             1 6 7
             1 7 2]
    x = p_sfc[:, 1]
    y = p_sfc[:, 2]
    np_sfc = size(p_sfc, 1)
    nt_sfc = size(t_sfc, 1)

    tplot(p_sfc, t_sfc)
    axis("equal")
    ylim(-1.1*r, 1.1*r)
    savefig("scratch/images/hex.png")
    println("scratch/images/hex.png")
    plt.close()

    # depths
    Hs = H.(p_sfc[:, 1], p_sfc[:, 2])

    # number of nodes in vertical
    nzs = Int64.(ceil.(Hs./h))

    # mapping from points to triangles:
    #   `p_to_tri[i]` is vector of cartesian indices pointing to where point `i` is in `t_sfc`
    p_to_tri = [findall(I -> i ∈ t_sfc[I], CartesianIndices(size(t_sfc))) for i=1:np_sfc]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    tri_to_p = [Int64[] for k=1:nt_sfc, i=1:3] # allocate

    # save middle stack
    stack = zeros(nzs[1])

    # add points to p, e, and tri_to_p
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>Int64[], "bot"=>Int64[])
    np = 0
    for i=1:np_sfc
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0]
        else
            z = -range(0, Hs[i], length=nz)
        end

        # add to p
        p[np+1:np+nz, :] = [x[i]*ones(nz)  y[i]*ones(nz)  z]
        if i == 1
            # save central stack
            stack[:] = z
        end

        # add to e
        e["sfc"] = [e["sfc"]; np + 1]
        if nz != 1
            e["bot"] = [e["bot"]; np + nz]
        end

        # add to tri_to_p
        for I ∈ p_to_tri[i]
            for j=np+1:np+nz
                push!(tri_to_p[I], j)
            end
        end

        # iterate
        np += nz
    end

    # setup shape functions and their integrals now since they're the same for each grid
    sf = ShapeFunctions(order=order, dim=3)
    sfi = ShapeFunctionIntegrals(sf, sf)

    # columnwise and global tessellation
    cols = Vector{FEGrid}(undef, nt_sfc)
    t = Matrix{Int64}(undef, 0, 4) 
    for k=1:nt_sfc
        # number of points in vertical for each vertex of sfc tri
        lens = length.(tri_to_p[k, :])

        # local p and e for column
        nodes_col = [tri_to_p[k, 1]; tri_to_p[k, 2]; tri_to_p[k, 3]]
        p_col = p[nodes_col, :]  
        e_sfc_col = [1, lens[1]+1, lens[1]+lens[2]+1]
        e_bot_col = [lens[1], lens[1]+lens[2], lens[1]+lens[2]+lens[3]]

        # start local t
        t_col = Matrix{Int64}(undef, 0, 4) 

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

            # add to t_col
            i_col = Int64.(indexin(ig, nodes_col))
            t_col = [t_col; i_col[tl]]

            # continue
            top = bot
        end

        # create e_col dictionary
        e_col = Dict("sfc"=>e_sfc_col, "bot"=>e_bot_col)

        # save column data
        cols[k] = FEGrid(order, p_col, t_col, e_col, sf, sfi)

        # remove from bot if in sfc
        cols[k].e["bot"] = cols[k].e["bot"][findall(i -> cols[k].e["bot"][i] ∉ cols[k].e["sfc"], 1:size(cols[k].e["bot"], 1))]
    end

    g = FEGrid(order, p, t, e)

    return cols, g, stack
end

function main()
    # Ekman
    ε² = 1

    # depth
    H(x, y) = 1 - x^2 - y^2

    # grid
    h = 0.05
    cols, g, stack = gen_hex(h, H, order=1)
    cols2, g2, stack2 = gen_hex(h, H, order=2)
    nz = size(stack, 1)

    # buoyancy
    b(x, y, z) = exp(z)*sin(x)*cos(y)
    bx(x, y, z) = exp(z)*cos(x)*cos(y)
    by(x, y, z) = -exp(z)*sin(x)*sin(y)
    b_cols = [b.(col.p[:, 1], col.p[:, 2], col.p[:, 3]) for col ∈ cols2]

    # buoyancy gradients
    bx_stack = zeros(2nz-2)
    by_stack = zeros(2nz-2)
    αs = [1/6, 1/8, 1/4, 1/6, 1/8, 1/6]
    for k ∈ eachindex(cols)
        b_col = FEField(b_cols[k], cols2[k])
        for j=1:nz-1
            k_tet = findfirst(k_tet -> j ∈ cols[k].t[k_tet, :] && j+1 ∈ cols[k].t[k_tet, :], 1:cols[k].nt)
            bx_stack[2j-1] += ∂x(b_col, [0, 0, stack[j]], k_tet)*αs[k]
            bx_stack[2j] += ∂x(b_col, [0, 0, stack[j+1]], k_tet)*αs[k]
            by_stack[2j-1] += ∂y(b_col, [0, 0, stack[j]], k_tet)*αs[k]
            by_stack[2j] += ∂y(b_col, [0, 0, stack[j+1]], k_tet)*αs[k]
        end
    end

    # transports
    Ux = 0
    Uy = 0

    # wind stress
    τx = 0
    τy = 0

    # solve
    sol = solve_baroclinic_1dfe(stack, bx_stack, by_stack, Ux, Uy, τx, τy, ε²)

    # unpack
    p = reshape(stack, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    e = Dict("bot"=>[nz], "sfc"=>[1])
    g = FEGrid(1, p, t, e)
    ωx = FEField(sol[1:nz], g)
    ωy = FEField(sol[nz+1:end], g)

    # fd sol 
    x = 0
    y = 0
    z = range(stack[end], stack[1], length=2^10)
    bx_a = bx.(x, y, z)
    by_a = by.(x, y, z)
    ωx_fd, ωy_fd = solve_baroclinic_1dfd(z, bx_a, by_a, Ux, Uy, ε²)

    # plot b grads
    fig, ax = subplots(1, figsize=(2.2, 3))
    ii = hcat(2:nz-1, 2:nz-1)[:]
    z_b = [stack[1]; stack[ii]; stack[end]]
    ax.plot(bx_stack, z_b, "o", ms=1, label=L"\partial_x b")
    ax.plot(by_stack, z_b, "o", ms=1, label=L"\partial_y b")
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
    ax.plot(ωx.values, ωx.g.p, label=L"\omega^x")
    ax.plot(ωy.values, ωy.g.p, label=L"\omega^y")
    ax.plot(ωx_fd, z, "k--", lw=0.5, label="FD Sol")
    ax.plot(ωy_fd, z, "k--", lw=0.5)
    ax.set_xlabel(L"\omega")
    ax.set_ylabel(L"z")
    ax.legend()
    savefig("scratch/images/omega.png")
    println("scratch/images/omega.png")
    plt.close()

    # error
    println(@sprintf("Max error ωx: %1.1e", maximum(x->isnan(x) ? -Inf : x, abs.(ωx.(z) - ωx_fd))))
    println(@sprintf("Max error ωy: %1.1e", maximum(x->isnan(x) ? -Inf : x, abs.(ωy.(z) - ωy_fd))))
end

main()
println("Done.")