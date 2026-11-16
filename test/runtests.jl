using Test
using nuPGCM
using Ferrite
using SparseArrays
using LinearAlgebra
using Printf
using JLD2
using Krylov

# internal assembly functions not part of the public API
using nuPGCM: build_M, build_Kₕ, build_Kᵥ, build_rhs_diff, build_rhs_flux, build_rhs_adv,
              collect_evolution_LHS!,
              build_A_inversion, build_A_visc!,
              build_B_inversion, build_f_wind,
              InversionLHSCache, refresh_A_cond!, update_A!,
              BlockTriangularPreconditioner,
              make_cell_values, make_facet_values,
              allocate_inversion_matrix, allocate_evolution_matrix,
              ν_eddy, get_n_dofs

# ---------------------------------------------------------------------------
# Shared test mesh
# ---------------------------------------------------------------------------
const MESH_FILE = joinpath(@__DIR__,
    "../meshes/channel_basin_flat_h1.00e-01_a5.00e-01.msh")

function ensure_test_mesh()
    isfile(MESH_FILE) && return
    @info "Generating test mesh..."
    include(joinpath(@__DIR__, "../meshes/channel_basin_flat.jl"))
    mesh_channel_basin_flat(0.1, 0.5)
end

ensure_test_mesh()
include("periodic_common.jl")

# ---------------------------------------------------------------------------
# Shared fixtures (created once; available in every included test file)
# ---------------------------------------------------------------------------
function make_fe_data()
    mesh = Mesh(MESH_FILE)
    return FEData(mesh;
        u_diri_tags  = ["bottom", "surface"],
        u_diri_masks = [(true,true,true), (false,false,true)],
        b_diri_tags  = ["surface"],
        b_diri_vals  = [x -> 0.0])
end

function make_params()
    Parameters(; ε=0.2, α=0.5, μϱ=10.0, N²=2.0, f=x->1.0, H=x->1.0)
end

function make_forcings()
    Forcings(1.0,
             x -> 1e-2 + 1e-3*x[3]^2,
             x -> 1e-2 + 1e-3*x[3]^2,
             x -> 0.0, x -> 0.0,
             SurfaceDirichletBC(x -> 0.0))
end

const FE_DATA  = make_fe_data()
const PARAMS   = make_params()
const FORCINGS = make_forcings()

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
@testset "nuPGCM Tests" begin
    include("test_meshes.jl")
    include("test_dofs.jl")
    include("test_spaces.jl")
    include("test_evolution.jl")
    include("test_inversion.jl")
    include("test_model.jl")
    include("test_periodic_box.jl")
    include("test_periodic_advection.jl")
    include("test_bowl.jl")
end
