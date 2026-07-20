@testset "FEData" begin
    fe_data = FE_DATA
    nu, np, nb = get_n_dofs(fe_data)

    @testset "DOF counts" begin
        @test nu > 0 && np > 0 && nb > 0

        # P2/P1: velocity is 3-component P2, so nu = 3 × (scalar P2 node count)
        @test nu == 3 * nb            # buoyancy also uses P2
        @test np < nb                 # P1 pressure < P2 buoyancy

        @test ndofs(fe_data.dh_up) == nu + np
        @test ndofs(fe_data.dh_b)  == nb

        @test fe_data.u_order == 2
        @test fe_data.b_order == 2
    end

    @testset "u/p DOF index sets" begin
        @test length(fe_data.u_dof_indices) == nu
        @test length(fe_data.p_dof_indices) == np
        @test isempty(intersect(fe_data.u_dof_indices, fe_data.p_dof_indices))
        @test sort(vcat(fe_data.u_dof_indices, fe_data.p_dof_indices)) ==
              collect(1:nu+np)        # u ∪ p covers the full dh_up range
    end

    @testset "Constraints" begin
        # ch_up holds velocity Dirichlet (bottom, surface z) + periodic + mean pressure
        @test length(fe_data.ch_up.prescribed_dofs) > 0
        @test length(fe_data.ch_up.prescribed_dofs) < nu + np

        # at least one pressure DOF is constrained in ch_up:
        # mean pressure AffineConstraint (non-periodic) or periodic image DOFs (periodic)
        @test !isempty(intersect(fe_data.ch_up.prescribed_dofs, fe_data.p_dof_indices))

        # ch_b holds surface Dirichlet + periodic
        @test length(fe_data.ch_b.prescribed_dofs) > 0
        @test length(fe_data.ch_b.prescribed_dofs) < nb
    end
end
