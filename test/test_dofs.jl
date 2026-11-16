@testset "FEData" begin
    fe_data = FE_DATA
    nu, np, nb = get_n_dofs(fe_data)

    @testset "DOF counts" begin
        @test nu == 3 * nb            # P2 velocity (3 components) and P2 buoyancy
        @test np < nb                 # P1 pressure
        @test ndofs(fe_data.dh_up) == nu + np
        @test ndofs(fe_data.dh_b)  == nb
        @test fe_data.u_order == 2
        @test fe_data.b_order == 2
    end

    @testset "u/p DOF index sets" begin
        @test length(fe_data.u_dof_indices) == nu
        @test length(fe_data.p_dof_indices) == np
        @test sort(vcat(fe_data.u_dof_indices, fe_data.p_dof_indices)) == collect(1:nu+np)
    end

    @testset "Constraints" begin
        prescribed_up = fe_data.ch_up.prescribed_dofs
        @test 0 < length(prescribed_up) < nu + np
        # pressure is constrained too: the gauge and the periodic seam
        @test !isempty(intersect(prescribed_up, fe_data.p_dof_indices))
        @test sort(fe_data.free_dofs) == setdiff(1:nu+np, prescribed_up)
        @test 0 < length(fe_data.ch_b.prescribed_dofs) < nb
    end
end
