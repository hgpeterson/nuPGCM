using nuPGCM
using HDF5
using PyPlot
using Printf
using ProgressMeter

include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_mesh(ifile; order)
    # load mesh of circle
    file = h5open(ifile, "r")
    p_sfc = read(file, "p")
    t_sfc = Int64.(read(file, "t"))
    e_sfc = Int64.(read(file, "e")[:, 1])
    close(file)
    x = p_sfc[:, 1]
    y = p_sfc[:, 2]
    np_sfc = size(p_sfc, 1)
    nt_sfc = size(t_sfc, 1)

    # mesh res
    emap, edges, bndix = all_edges(t_sfc)
    h = 1/size(edges, 1)*sum(norm(p_sfc[edges[i, 1], :] - p_sfc[edges[i, 2], :]) for i in axes(edges, 1))

    # mapping from points to triangles:
    #   `p_to_tri[i]` is vector of cartesian indices pointing to where point `i` is in `t_sfc`
    p_to_tri = [findall(I -> i ∈ t_sfc[I], CartesianIndices(size(t_sfc))) for i=1:np_sfc]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    tri_to_p = [Int64[] for k=1:nt_sfc, i=1:3] # allocate

    # stacks
    stacks = Matrix{Matrix{Float64}}(undef, nt_sfc, 3)

    # add points to p, e, and tri_to_p
    nzs = Int64[i ∈ e_sfc ? 1 : ceil(H(x[i], y[i])/h) for i=1:np_sfc]
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>Int64[], "bot"=>Int64[])
    np = 0
    for i=1:np_sfc
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0]
        else
            z = -range(0, H(x[i], y[i]), length=nz)
        end

        # add to p
        p[np+1:np+nz, :] = [x[i]*ones(nz)  y[i]*ones(nz)  z]
        for I ∈ p_to_tri[i]
            stacks[I] = [x[i]*ones(nz)  y[i]*ones(nz)  z]
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

    return cols, g, stacks
end

# params
ε² = 1
H(x, y) = 1 - x^2 - y^2
Hx(x, y) = -2*x
Hy(x, y) = -2*y
# Ux(x, y) = 0
# Uy(x, y) = 0
# Ux(x, y) = H(x, y)^2
# Uy(x, y) = H(x, y)^2
# b(x, y, z) = 0
# bx(x, y, z) = 0
# by(x, y, z) = 0
# b(x, y, z) = x
# bx(x, y, z) = 1
# by(x, y, z) = 0
# b(x, y, z) = x^3 + y^3
# bx(x, y, z) = 3x^2
# by(x, y, z) = 3y^2
# δ = 0.1
# b(x, y, z) = z + δ*exp(-(z + H(x, y))/δ)
# bx(x, y, z) = -Hx(x, y)*exp(-(z + H(x, y))/δ)
# by(x, y, z) = -Hy(x, y)*exp(-(z + H(x, y))/δ)

# analytical solution
ωx_a(x, y, z) = z*exp(z)*cos(x)*sin(y)
ωy_a(x, y, z) = z*exp(z)*cos(y)*sin(x)
Ux(x, y) = -(2 - exp(-H(x, y))*(2 + 2*H(x, y) + H(x, y)^2))*cos(y)*sin(x)
Uy(x, y) =  (2 - exp(-H(x, y))*(2 + 2*H(x, y) + H(x, y)^2))*cos(x)*sin(y)
τx(x, y) = ωy_a(x, y, 0)
τy(x, y) = -ωx_a(x, y, 0)
b(x, y, z) = -exp(z)*(z*sin(x)*sin(y) - ε²*(z + 2)*cos(x)*cos(y))

grid
cols, g, stacks = gen_mesh("meshes/circle/mesh1.h5", order=1)
nzs = [size(stacks[k, i], 1) for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)]
println("ncols = ", size(cols, 1))

# b, bx, by, Ux, Uy in each column
b_cols = [b.(col.p[:, 1], col.p[:, 2], col.p[:, 3]) for col ∈ cols]
Ux_stacks = [Ux.(stacks[k, i][1, 1], stacks[k, i][1, 2]) for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)]
Uy_stacks = [Uy.(stacks[k, i][1, 1], stacks[k, i][1, 2]) for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)]
τx_stacks = [τx.(stacks[k, i][1, 1], stacks[k, i][1, 2]) for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)]
τy_stacks = [τy.(stacks[k, i][1, 1], stacks[k, i][1, 2]) for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)]
bx_stacks = Matrix{Vector{Float64}}(undef, size(stacks, 1), size(stacks, 2))
by_stacks = Matrix{Vector{Float64}}(undef, size(stacks, 1), size(stacks, 2))
for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)
    b_col = FEField(b_cols[k], cols[k])
    bx_stacks[k, i] = zeros(nzs[k, i]-1)
    by_stacks[k, i] = zeros(nzs[k, i]-1)
    for j=1:nzs[k, i]-1
        n = i == 1 ? 0 : sum(nzs[k, i_stack] for i_stack=1:i-1)
        k_tet = findfirst(k_tet -> n+j ∈ cols[k].t[k_tet, :] && n+j+1 ∈ cols[k].t[k_tet, :], 1:cols[k].nt)
        bx_stacks[k, i][j] = ∂x(b_col, [0, 0, 0], k_tet)
        by_stacks[k, i][j] = ∂y(b_col, [0, 0, 0], k_tet)
    end
end

# solve
sols = [nzs[k, i] == 1 ? [0.0, 0.0] : solve_baroclinic_1dfe(stacks[k, i][:, 3], bx_stacks[k, i], by_stacks[k, i], Ux_stacks[k, i], Uy_stacks[k, i], τx_stacks[k, i], τy_stacks[k, i], ε²) for k ∈ axes(stacks, 1), i ∈ axes(stacks, 2)]

plot_3D()

println("Done.")