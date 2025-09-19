using Test
using nuPGCM.FiniteElements

@testset "P1" begin
    space = P1()
    @test FiniteElements.φ(Line(), space, -1.0) ≈ [1, 0]
    @test FiniteElements.φ(Line(), space,  1.0) ≈ [0, 1]
    @test FiniteElements.φ(Triangle(), space, [0.0, 0.0]) ≈ [1, 0, 0]
    @test FiniteElements.φ(Triangle(), space, [1.0, 0.0]) ≈ [0, 1, 0]
    @test FiniteElements.φ(Triangle(), space, [0.0, 1.0]) ≈ [0, 0, 1]
end

@testset "P2" begin
    space = P2()
    @test FiniteElements.φ(Line(), space, -1.0) ≈ [1, 0, 0]
    @test FiniteElements.φ(Line(), space,  1.0) ≈ [0, 1, 0]
    @test FiniteElements.φ(Line(), space,  0.0) ≈ [0, 0, 1]
    @test FiniteElements.φ(Triangle(), space, [0.0, 0.0]) ≈ [1, 0, 0, 0, 0, 0]
    @test FiniteElements.φ(Triangle(), space, [1.0, 0.0]) ≈ [0, 1, 0, 0, 0, 0]
    @test FiniteElements.φ(Triangle(), space, [0.0, 1.0]) ≈ [0, 0, 1, 0, 0, 0]
    @test FiniteElements.φ(Triangle(), space, [0.5, 0.0]) ≈ [0, 0, 0, 1, 0, 0]
    @test FiniteElements.φ(Triangle(), space, [0.5, 0.5]) ≈ [0, 0, 0, 0, 1, 0]
    @test FiniteElements.φ(Triangle(), space, [0.0, 0.5]) ≈ [0, 0, 0, 0, 0, 1]
end
