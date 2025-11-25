using nuPGCM
using Test

@testset "nuPGCM Tests" begin

    @testset "Unit tests" begin
        include("utils_tests.jl")
    end

    @testset "End-to-end tests" begin
        include("bowl_mixing_tests.jl")
        # include("bowl_wind_tests.jl")
        # include("bowl_surface_flux_tests.jl")
    end

end