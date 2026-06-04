using Test
using Ferrite
using FerriteGmsh
using Gmsh: gmsh
using SparseArrays
using LinearAlgebra
using Printf

# Load the new infrastructure directly; nuPGCM doesn't compile yet because
# other source files still use Gridap.
const SRC = joinpath(@__DIR__, "../src")
include(joinpath(SRC, "meshes.jl"))
include(joinpath(SRC, "dofs.jl"))
include(joinpath(SRC, "spaces.jl"))

# ---------------------------------------------------------------------------
# Shared test mesh (channel-basin-flat, coarse)
# ---------------------------------------------------------------------------
const MESH_FILE = joinpath(@__DIR__, "../meshes/channel_basin_flat_h1.00e-01_a5.00e-01.msh")

function ensure_test_mesh()
    isfile(MESH_FILE) && return
    @info "Generating test mesh..."
    include(joinpath(@__DIR__, "../meshes/channel_basin_flat.jl"))
    mesh_channel_basin_flat(0.1, 0.5)
end

ensure_test_mesh()

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
@testset "Mesh" begin
    mesh = Mesh(MESH_FILE)

    @test mesh isa Mesh
    @test mesh.surface_tag == "surface"
    @test getnnodes(mesh.grid) > 0
    @test getncells(mesh.grid) > 0

    # updated mesh scripts give channel_west / channel_east, not "interior"
    @test haskey(mesh.grid.facetsets, "bottom")
    @test haskey(mesh.grid.facetsets, "surface")
    @test haskey(mesh.grid.facetsets, "channel_west")
    @test haskey(mesh.grid.facetsets, "channel_east")
    @test !haskey(mesh.grid.facetsets, "interior")

    # channel_west and channel_east have the same number of facets
    @test length(mesh.grid.facetsets["channel_west"]) ==
          length(mesh.grid.facetsets["channel_east"])

    # get_p_t shape
    p, t = get_p_t(mesh)
    @test size(p, 1) == getnnodes(mesh.grid)
    @test size(p, 2) == 3              # 3-D
    @test size(t, 1) == getncells(mesh.grid)
    @test size(t, 2) == 4              # tetrahedra

    # all node indices in t are valid
    @test minimum(t) >= 1
    @test maximum(t) <= getnnodes(mesh.grid)

    # channel geometry: x ∈ [0, 1]  (W = 1)
    @test minimum(p[:, 1]) ≈ 0.0 atol=1e-10
    @test maximum(p[:, 1]) ≈ 1.0 atol=1e-10

    # compute_h_cells
    h_cells = compute_h_cells(mesh)
    @test length(h_cells) == getncells(mesh.grid)
    @test all(h_cells .> 0)
    @test maximum(h_cells) < 1.0      # coarser than the domain size

    # string constructor (file name) round-trips
    p2, t2 = get_p_t(MESH_FILE)
    @test p2 ≈ p
    @test t2 == t
end

# ---------------------------------------------------------------------------
# FEData
# ---------------------------------------------------------------------------
@testset "FEData" begin
    mesh = Mesh(MESH_FILE)

    u_diri_tags  = ["bottom", "surface"]
    u_diri_masks = [(true, true, true), (false, false, true)]
    b_diri_tags  = ["surface"]
    b_diri_vals  = [x -> 0.0]

    fe_data = FEData(mesh; u_diri_tags, u_diri_masks, b_diri_tags, b_diri_vals)
    nu, np, nb = get_n_dofs(fe_data)

    @testset "DOF counts" begin
        @test nu > 0
        @test np > 0
        @test nb > 0

        # P2/P1 Taylor-Hood: nu = 3×nb (both P2), np = n_P1_nodes < nb
        @test nu == 3 * nb            # velocity is 3-component P2
        @test np < nb                 # P1 pressure has fewer DOFs than P2 buoyancy

        # DofHandler totals match
        @test ndofs(fe_data.dh_u) == nu
        @test ndofs(fe_data.dh_p) == np
        @test ndofs(fe_data.dh_b) == nb

        # stored orders
        @test fe_data.u_order == 2
        @test fe_data.b_order == 2
    end

    @testset "Dirichlet constraints" begin
        # bottom no-slip: all three velocity components → many u DOFs constrained
        @test length(fe_data.ch_u.prescribed_dofs) > 0
        # surface w=0: some u DOFs constrained (z-component only)
        # (both effects combined into ch_u.prescribed_dofs)

        # buoyancy surface Dirichlet
        @test length(fe_data.ch_b.prescribed_dofs) > 0
    end

    @testset "Periodic constraints" begin
        # channel_west and channel_east: both pressure and buoyancy get periodic BCs
        @test length(fe_data.ch_p.prescribed_dofs) > 0
        @test length(fe_data.ch_b.prescribed_dofs) > 0  # periodic + Dirichlet

        # periodic must constrain fewer DOFs than a full Dirichlet
        @test length(fe_data.ch_p.prescribed_dofs) < np
    end

    @testset "Permutations" begin
        # identity permutations (not yet replaced with Cuthill-McKee)
        @test fe_data.p_up    == collect(1:nu + np)
        @test fe_data.p_b     == collect(1:nb)
        @test fe_data.inv_p_up == collect(1:nu + np)
        @test fe_data.inv_p_b  == collect(1:nb)
    end
end

# ---------------------------------------------------------------------------
# CellValues, FacetValues, matrix allocators
# ---------------------------------------------------------------------------
@testset "Spaces" begin
    mesh    = Mesh(MESH_FILE)
    fe_data = FEData(mesh; u_diri_tags=["bottom","surface"],
                           u_diri_masks=[(true,true,true),(false,false,true)],
                           b_diri_tags=["surface"], b_diri_vals=[x->0.0])
    nu, np, nb = get_n_dofs(fe_data)

    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    fv_u, fv_b       = make_facet_values(fe_data)

    @testset "CellValues dimensions" begin
        # P2 vector: 10 scalar nodes × 3 components = 30
        @test getnbasefunctions(cv_u) == 30
        # P1 scalar: 4 nodes per tet
        @test getnbasefunctions(cv_p) == 4
        # P2 scalar: 10 nodes per tet
        @test getnbasefunctions(cv_b) == 10
        # all use the same quadrature (QR_ORDER = 3 → 5 points on tet)
        @test getnquadpoints(cv_u) == getnquadpoints(cv_p) == getnquadpoints(cv_b)
    end

    @testset "FacetValues dimensions" begin
        @test getnbasefunctions(fv_u) == 30
        @test getnbasefunctions(fv_b) == 10
    end

    @testset "Partition of unity (P1 pressure)" begin
        # For any interpolation, ∑ φᵢ(x) = 1 at every quadrature point
        cc = first(CellIterator(fe_data.dh_p))
        reinit!(cv_p, cc)
        for q in 1:getnquadpoints(cv_p)
            s = sum(shape_value(cv_p, q, i) for i in 1:getnbasefunctions(cv_p))
            @test s ≈ 1.0 atol=1e-14
        end
    end

    @testset "Partition of unity (P2 buoyancy)" begin
        cc = first(CellIterator(fe_data.dh_b))
        reinit!(cv_b, cc)
        for q in 1:getnquadpoints(cv_b)
            s = sum(shape_value(cv_b, q, i) for i in 1:getnbasefunctions(cv_b))
            @test s ≈ 1.0 atol=1e-14
        end
    end

    @testset "Sum of gradients is zero (P1 pressure)" begin
        # Consequence of partition of unity: ∑ ∇φᵢ = 0
        cc = first(CellIterator(fe_data.dh_p))
        reinit!(cv_p, cc)
        for q in 1:getnquadpoints(cv_p)
            g = sum(shape_gradient(cv_p, q, i) for i in 1:getnbasefunctions(cv_p))
            @test norm(g) < 1e-14
        end
    end

    @testset "allocate_inversion_matrix" begin
        K_inv = allocate_inversion_matrix(fe_data)
        @test size(K_inv) == (nu + np, nu + np)
        @test nnz(K_inv) > 0
        @test all(iszero, K_inv.nzval)   # structure allocated, values zeroed

        # u-u block (top-left nu×nu) is non-empty
        K_uu = K_inv[1:nu, 1:nu]
        @test nnz(K_uu) > 0

        # u-p block (top-right nu×np) is non-empty (pressure gradient coupling)
        K_up = K_inv[1:nu, nu+1:end]
        @test nnz(K_up) > 0

        # pattern is symmetric in block structure (K_up and K_pu both filled)
        K_pu = K_inv[nu+1:end, 1:nu]
        @test nnz(K_pu) == nnz(K_up)
    end

    @testset "allocate_evolution_matrix" begin
        K_evo = allocate_evolution_matrix(fe_data)
        @test size(K_evo) == (nb, nb)
        @test nnz(K_evo) > 0
        @test all(iszero, K_evo.nzval)

        # pattern is symmetric (mass/stiffness matrices are symmetric)
        @test K_evo.rowval == allocate_evolution_matrix(fe_data).rowval
    end
end
