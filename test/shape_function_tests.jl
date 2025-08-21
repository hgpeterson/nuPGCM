using Test
using nuPGCM.FiniteElements

@testset "Lagrangian elements" begin
    sf = Lagrange(1)
    @test FiniteElements.φ(Line(), sf, -1.0) ≈ [1, 0]
    @test FiniteElements.φ(Line(), sf,  1.0) ≈ [0, 1]
    @test FiniteElements.φ(Triangle(), sf, [0.0, 0.0]) ≈ [1, 0, 0]
    @test FiniteElements.φ(Triangle(), sf, [1.0, 0.0]) ≈ [0, 1, 0]
    @test FiniteElements.φ(Triangle(), sf, [0.0, 1.0]) ≈ [0, 0, 1]

    sf = Lagrange(2)
    @test FiniteElements.φ(Line(), sf, -1.0) ≈ [1, 0, 0]
    @test FiniteElements.φ(Line(), sf,  1.0) ≈ [0, 1, 0]
    @test FiniteElements.φ(Line(), sf,  0.0) ≈ [0, 0, 1]
    @test FiniteElements.φ(Triangle(), sf, [0.0, 0.0]) ≈ [1, 0, 0, 0, 0, 0]
    @test FiniteElements.φ(Triangle(), sf, [1.0, 0.0]) ≈ [0, 1, 0, 0, 0, 0]
    @test FiniteElements.φ(Triangle(), sf, [0.0, 1.0]) ≈ [0, 0, 1, 0, 0, 0]
    @test FiniteElements.φ(Triangle(), sf, [0.5, 0.0]) ≈ [0, 0, 0, 1, 0, 0]
    @test FiniteElements.φ(Triangle(), sf, [0.5, 0.5]) ≈ [0, 0, 0, 0, 1, 0]
    @test FiniteElements.φ(Triangle(), sf, [0.0, 0.5]) ≈ [0, 0, 0, 0, 0, 1]
end
