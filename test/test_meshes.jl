@testset "Mesh" begin
    mesh = FE_DATA.mesh

    @test mesh isa Mesh
    @test mesh.surface_tag == "surface"
    @test getnnodes(mesh.grid) > 0
    @test getncells(mesh.grid) > 0

    # updated mesh scripts: channel_west / channel_east instead of "interior"
    @test haskey(mesh.grid.facetsets, "bottom")
    @test haskey(mesh.grid.facetsets, "surface")
    @test haskey(mesh.grid.facetsets, "channel_west")
    @test haskey(mesh.grid.facetsets, "channel_east")
    @test !haskey(mesh.grid.facetsets, "interior")

    # periodic walls are matched in size
    @test length(mesh.grid.facetsets["channel_west"]) ==
          length(mesh.grid.facetsets["channel_east"])

    # get_p_t
    p, t = get_p_t(mesh)
    @test size(p, 1) == getnnodes(mesh.grid)
    @test size(p, 2) == 3
    @test size(t, 1) == getncells(mesh.grid)
    @test size(t, 2) == 4              # tetrahedra

    @test minimum(t) >= 1
    @test maximum(t) <= getnnodes(mesh.grid)

    # channel occupies x ∈ [0, 1]  (W = 1)
    @test minimum(p[:, 1]) ≈ 0.0 atol=1e-10
    @test maximum(p[:, 1]) ≈ 1.0 atol=1e-10

    # compute_h_cells
    h = compute_h_cells(mesh)
    @test length(h) == getncells(mesh.grid)
    @test all(h .> 0)
    @test maximum(h) < 1.0

    # string-path constructor round-trips
    p2, t2 = get_p_t(MESH_FILE)
    @test p2 ≈ p
    @test t2 == t
end
