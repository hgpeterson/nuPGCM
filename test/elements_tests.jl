using Test
using nuPGCM.FiniteElements

@testset "Element types" begin
    @test FiniteElements.get_element_type(0) isa Point
    @test FiniteElements.get_element_type(1) isa Line
    @test FiniteElements.get_element_type(2) isa Triangle
    @test FiniteElements.get_element_type(3) isa Tetrahedron
    @test_throws ArgumentError FiniteElements.get_element_type(-1)
    @test_throws ArgumentError FiniteElements.get_element_type(4)
end

@testset "Reference elements" begin
    @test FiniteElements.reference_element(Point()) == [0.0]
    @test FiniteElements.reference_element(Line()) == [-1.0, 1.0]
    @test FiniteElements.reference_element(Triangle()) == [0.0 0.0; 1.0 0.0; 0.0 1.0]
    @test FiniteElements.reference_element(Tetrahedron()) == [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
end

@testset "Transformation matrix/vector" begin
    vertices_line = [-1.0 0.0 0.0; 1.0 0.0 0.0]
    A, b = FiniteElements.transformation_matrix_vector(Line(), vertices_line)
    @test A ≈ 1.0
    @test b ≈ [0.0, 0.0, 0.0]

    vertices_triangle = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
    A, b = FiniteElements.transformation_matrix_vector(Triangle(), vertices_triangle)
    @test A ≈ [1.0 0.0; 0.0 1.0]
    @test b ≈ [0.0, 0.0]

    vertices_tetrahedron = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    A, b = FiniteElements.transformation_matrix_vector(Tetrahedron(), vertices_tetrahedron)
    @test A ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @test b ≈ [0.0, 0.0, 0.0]
end