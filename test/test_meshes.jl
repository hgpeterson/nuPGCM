@testset "Mesh" begin
    mesh = FE_DATA.mesh

    @test mesh.surface_tag == "surface"
    for tag in ("bottom", "surface", "channel_west", "channel_east")
        @test haskey(mesh.grid.facetsets, tag)
    end
    # periodic walls are matched in size
    @test length(mesh.grid.facetsets["channel_west"]) ==
          length(mesh.grid.facetsets["channel_east"])

    p, t = get_p_t(mesh)
    @test size(p) == (getnnodes(mesh.grid), 3)
    @test size(t) == (getncells(mesh.grid), 4)   # tetrahedra
    @test extrema(t) == (1, getnnodes(mesh.grid))
    @test minimum(p[:, 1]) ≈ 0.0 atol=1e-10      # channel occupies x ∈ [0, 1]
    @test maximum(p[:, 1]) ≈ 1.0 atol=1e-10

    h = compute_h_cells(mesh)
    @test length(h) == getncells(mesh.grid)
    @test all(0 .< h .< 1)

    # string-path method agrees
    p2, t2 = get_p_t(MESH_FILE)
    @test p2 ≈ p
    @test t2 == t
end
