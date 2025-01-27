@testset "Utilities Testset" begin
    @test chebyshev_nodes(5) ≈ [-1.0, -0.8535533905932737, -0.5, -0.14644660940672627, 0.0]
    @test trapz([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) ≈ 4.0
    @test hrs_mins_secs(3661) == (1, 1, 1)
    @test nan_max([1.0, 2.0, NaN, 3.0]) == 3.0
    @test nan_min([1.0, 2.0, NaN, 3.0]) == 1.0
    @test sci_notation(0.00079) == "7.9 \\times 10^{-4}"
end