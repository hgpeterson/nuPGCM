@testset "Spaces" begin
    fe_data = FE_DATA
    nu, np, nb = get_n_dofs(fe_data)

    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    fv_u, fv_b       = make_facet_values(fe_data)

    @testset "element orders" begin
        @test getnbasefunctions(cv_u) == 30   # P2 vector: 10 nodes × 3
        @test getnbasefunctions(cv_p) == 4    # P1
        @test getnbasefunctions(cv_b) == 10   # P2 scalar
        @test getnquadpoints(cv_u) == getnquadpoints(cv_p) == getnquadpoints(cv_b)
        @test getnbasefunctions(fv_u) == 30
        @test getnbasefunctions(fv_b) == 10
    end

    # the constraint-augmented patterns must contain the standard ones
    @testset "allocate_inversion_matrix" begin
        K = allocate_inversion_matrix(fe_data)
        @test size(K) == (nu+np, nu+np)
        @test all(iszero, K.nzval)
        @test nnz(K) >= nnz(allocate_matrix(fe_data.dh_up))
    end

    @testset "allocate_evolution_matrix" begin
        K = allocate_evolution_matrix(fe_data)
        @test size(K) == (nb, nb)
        @test all(iszero, K.nzval)
        @test nnz(K) >= nnz(allocate_matrix(fe_data.dh_b))
    end
end
