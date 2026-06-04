using Test

@testset "nuPGCM Tests" begin
    # mesh loading, DofHandlers, CellValues, matrix allocators
    include("test_ferrite_infra.jl")

    # Bowl regression tests (re-enabled once assembly and model loop are
    # rewritten in Ferrite)
    # include("bowl_mixing_tests.jl")
    # include("bowl_dirichlet_tests.jl")
    # include("bowl_wind_tests.jl")
    # include("bowl_surface_flux_tests.jl")
end
