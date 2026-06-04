@testset "Spaces" begin
    fe_data = FE_DATA
    nu, np, nb = get_n_dofs(fe_data)

    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    fv_u, fv_b       = make_facet_values(fe_data)

    @testset "CellValues dimensions" begin
        @test getnbasefunctions(cv_u) == 30   # P2 vec: 10 nodes × 3
        @test getnbasefunctions(cv_p) == 4    # P1: 4 nodes per tet
        @test getnbasefunctions(cv_b) == 10   # P2 scalar
        @test getnquadpoints(cv_u) == getnquadpoints(cv_p) == getnquadpoints(cv_b)
    end

    @testset "FacetValues dimensions" begin
        @test getnbasefunctions(fv_u) == 30
        @test getnbasefunctions(fv_b) == 10
    end

    @testset "Partition of unity (P1 pressure)" begin
        cc = first(CellIterator(fe_data.dh_up))
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
        cc = first(CellIterator(fe_data.dh_up))
        reinit!(cv_p, cc)
        for q in 1:getnquadpoints(cv_p)
            g = sum(shape_gradient(cv_p, q, i) for i in 1:getnbasefunctions(cv_p))
            @test norm(g) < 1e-14
        end
    end

    @testset "allocate_inversion_matrix" begin
        K = allocate_inversion_matrix(fe_data)
        @test size(K) == (nu+np, nu+np)
        @test nnz(K) > 0
        @test all(iszero, K.nzval)

        # BC-augmented pattern has more entries than the standard one
        K_std = allocate_matrix(fe_data.dh_up)
        @test nnz(K) >= nnz(K_std)

        # two allocations give identical structural patterns
        @test K.rowval == allocate_inversion_matrix(fe_data).rowval
    end

    @testset "allocate_evolution_matrix" begin
        K = allocate_evolution_matrix(fe_data)
        @test size(K) == (nb, nb)
        @test nnz(K) > 0
        @test all(iszero, K.nzval)

        K_std = allocate_matrix(fe_data.dh_b)
        @test nnz(K) >= nnz(K_std)

        @test K.rowval == allocate_evolution_matrix(fe_data).rowval
    end
end
