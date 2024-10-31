@testset "Utilities Testset" begin
    @test nuPGCM.chebyshev_nodes(5) ≈ [-1.0, -0.7071067811865476, 0.0, 0.7071067811865476, 1.0]
    @test nuPGCM.trapz([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) ≈ 4.0
    @test nuPGCM.hrs_mins_secs(3661) == (1, 1, 1)
    @test nuPGCM.nan_max([1.0, 2.0, NaN, 3.0]) == 3.0
    @test nuPGCM.nan_min([1.0, 2.0, NaN, 3.0]) == 1.0
    @test nuPGCM.sci_notation(0.00079) == "7.9 \\times 10^{-4}"
end