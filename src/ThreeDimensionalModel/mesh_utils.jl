function gen_3D_valign_mesh(geo, nref, H; chebyshev=false, tessellate=nothing)
    # surface mesh
    g_sfc = H.g

    # will we need to tessellate?
    if tessellate === nothing
        tessellate = !isfile("meshes/$geo/t_col_$(nref)_1.h5")
        # tessellate = true
    end

    # x and y for convenience
    x = g_sfc.p[:, 1]
    y = g_sfc.p[:, 2]

    # mesh res
    emap, edges, bndix = all_edges(g_sfc.t)
    h = 1/size(edges, 1)*sum(norm(g_sfc.p[edges[i, 1], :] - g_sfc.p[edges[i, 2], :]) for i in axes(edges, 1))

    # mapping from points to triangles:
    #   `p_to_tri[i]` is vector of cartesian indices pointing to where point `i` is in `g_sfc.t`
    p_to_tri = [findall(I -> i ∈ g_sfc.t[I], CartesianIndices(size(g_sfc.t))) for i=1:g_sfc.np]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    tri_to_p = [Int64[] for k=1:g_sfc.nt, i=1:3] # allocate

    # z_cols
    z_cols = Vector{Vector{Float64}}(undef, g_sfc.np)

    # add points to p, e, and tri_to_p
    nzs = Int64[i ∈ g_sfc.e["bdy"] ? 1 : ceil(H[i]/h) for i=1:g_sfc.np]
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>Int64[], "bot"=>Int64[])
    np = 0
    for i=1:g_sfc.np
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0]
        else
            if chebyshev
                z = -H[i]*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
            else
                z = range(-H[i], 0, length=nz)
            end
        end

        # add to p
        p[np+1:np+nz, :] = [x[i]*ones(nz)  y[i]*ones(nz)  z]
        z_cols[i] = z

        # add to e
        e["bot"] = [e["bot"]; np + 1]
        e["sfc"] = [e["sfc"]; np + nz]

        # add to tri_to_p
        for I ∈ p_to_tri[i]
            for j=np+1:np+nz
                push!(tri_to_p[I], j)
            end
        end

        # iterate
        np += nz
    end

    # size of each column
    nzs = [size(z, 1) for z ∈ z_cols]

    # setup shape functions and their integrals now since they're the same for each grid
    sf = ShapeFunctions(order=1, dim=3)
    sfi = ShapeFunctionIntegrals(sf, sf)

    # columnwise and global tessellation
    g_cols = Vector{Grid}(undef, g_sfc.nt)
    t = Matrix{Int64}(undef, 0, 4) 
    @showprogress "Generating columns..." for k=1:g_sfc.nt
        # number of points in vertical for each vertex of sfc tri
        lens = length.(tri_to_p[k, :])

        # local p and e for col
        nodes_col = [tri_to_p[k, 1]; tri_to_p[k, 2]; tri_to_p[k, 3]]
        p_col = p[nodes_col, :]  
        e_bot_col = [1, lens[1]+1, lens[1]+lens[2]+1]
        e_sfc_col = [lens[1], lens[1]+lens[2], lens[1]+lens[2]+lens[3]]

        # either compute or load t for col
        if tessellate
            t_col = generate_t_col(geo, nref, k, p, tri_to_p, lens, nodes_col)
        else
            t_col = load_t_col(geo, nref, k)
        end

        # add to global t
        t = [t; nodes_col[t_col]]

        # create e_col dictionary
        e_col = Dict("sfc"=>e_sfc_col, "bot"=>e_bot_col)

        # save column data
        g_cols[k] = Grid(1, p_col, t_col, e_col, sf, sfi)
    end

    g = Grid(1, p, t, e)

    return g, g_cols, z_cols, nzs, p_to_tri
end

function generate_t_col(geo, nref, k, p, tri_to_p, lens, nodes_col)
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

        # add to t_col
        i_col = Int64.(indexin(ig, nodes_col))
        t_col = [t_col; i_col[tl]]

        # continue
        top = bot
    end

    save_t_col(geo, nref, k, t_col)

    return t_col
end

function save_t_col(geo, nref, k, t_col)
    h5open("meshes/$geo/t_col_$(nref)_$k.h5", "w") do file
        write(file, "t_col", t_col)
    end
end

function load_t_col(geo, nref, k)
    file = h5open("meshes/$geo/t_col_$(nref)_$k.h5", "r")
    t_col = read(file, "t_col")
    close(file)
    return t_col
end
