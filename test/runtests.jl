using nuPGCM
using Test

@testset "nuPGCM Tests" begin
    include("bowl_mixing_tests.jl")
    include("bowl_dirichlet_tests.jl")
    include("bowl_wind_tests.jl")
    include("bowl_surface_flux_tests.jl")
end