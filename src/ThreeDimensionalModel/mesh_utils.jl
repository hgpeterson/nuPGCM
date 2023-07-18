function generate_wedge_cols(g_sfc1, g_sfc2; nσ=0, chebyshev=false)
    if nσ == 0
        # by default, compute nσ to roughly match surface mesh
        emap, edges, bndix = all_edges(g_sfc1.t)
        h = 1/size(edges, 1)*sum(norm(g_sfc1.p[edges[i, 1], :] - g_sfc1.p[edges[i, 2], :]) for i ∈ axes(edges, 1))
        nσ = Int64(round(1/h)) + 1
    end

    # σ coordinates for node grids
    if chebyshev
        σ = -(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2
    else
        σ = -1:1/(nσ-1):0
    end

    # node points on second order grid
    p2 = hcat(repeat(g_sfc2.p, inner=(nσ, 1)), repeat(σ, g_sfc2.np))
    np2 = g_sfc2.np*nσ

    # node connectivities on second order grid
    t2 = zeros(Int64, g_sfc2.nt*(nσ - 1), 12)
    for k_sfc=1:g_sfc2.nt
        for j=1:nσ-1
            t2[(k_sfc-1)*(nσ-1)+j, 1:3]   = nσ*(g_sfc2.t[k_sfc, 1:3] .- 1) .+ j
            t2[(k_sfc-1)*(nσ-1)+j, 4:6]   = nσ*(g_sfc2.t[k_sfc, 1:3] .- 1) .+ j .+ 1
            t2[(k_sfc-1)*(nσ-1)+j, 7:9]   = nσ*(g_sfc2.t[k_sfc, 4:6] .- 1) .+ j
            t2[(k_sfc-1)*(nσ-1)+j, 10:12] = nσ*(g_sfc2.t[k_sfc, 4:6] .- 1) .+ j .+ 1
        end
    end

    # boundaries on second order grid
    e2 = Dict("sfc"=>collect(nσ:nσ:np2), "bot"=>collect(1:nσ:np2-nσ+1))

    # second order grid
    g2 = Grid(2, p2, t2, e2)

    # first order grid
    # np1 = g_sfc1.np*nσ
    # e1 = Dict("sfc"=>collect(nσ:nσ:np1), "bot"=>collect(1:nσ:np1-nσ+1))
    # g1 = Grid(1, p2[1:np1, :], t2[:, 1:6], e1)
    g1 = Grid(1, g2)

    H = [1 - g1.p[i, 1]^2 - g1.p[i, 2]^2 for i=1:g1.np]
    pz = copy(g1.p)
    pz[:, 3] .*= H
    vtk_grid("$out_folder/mesh.vtu", pz', [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[k, :]) for k ∈ axes(g1.t, 1)]) do vtk end
    println("$out_folder/mesh.vtu")

    return g1, g2
end