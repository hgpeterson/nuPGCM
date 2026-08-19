module nuPGCM

using Ferrite
using WriteVTK
using FerriteGmsh
using CuthillMcKee
using JLD2
using LinearAlgebra
using SparseArrays
using Krylov
using KrylovPreconditioners
using PyPlot
using Printf

# unit vectors (Ferrite uses Tensors.Vec, re-exported by Ferrite)
const x⃗ = Vec{3, Float64}((1.0, 0.0, 0.0))
const y⃗ = Vec{3, Float64}((0.0, 1.0, 0.0))
const z⃗ = Vec{3, Float64}((0.0, 0.0, 1.0))

# directory where the output files will be saved
global out_dir = "."

"""
    set_out_dir!(dir)

Set the output directory where the results will be saved. The function creates 
the directory and two subdirectory:
- `dir`/images for plots, and 
- `dir`/data for data files.
"""
function set_out_dir!(dir)
    if out_dir != dir
        global out_dir = dir
        @info "Output directory set to '$dir'"
    end

    if !isdir(out_dir)
        @info "Creating directory '$out_dir'"
        mkdir(out_dir)
    end
    if !isdir("$out_dir/images")
        @info "Creating subdirectory '$out_dir/images'"
        mkdir("$out_dir/images")
    end
    if !isdir("$out_dir/data")
        @info "Creating subdirectory '$out_dir/data'"
        mkdir("$out_dir/data")
    end
end

# bool to turn on/off printing timings (default off)
const ENABLE_TIMING = Ref(false)

"""
    @ctime "description" expr

Conditionally run `@time` if `ENABLE_TIMING` is `true`.
"""
macro ctime(label, expr)
    quote
        if $ENABLE_TIMING[]
            @time $label $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

# include all the module code
include("architectures.jl")
include("utils.jl")
include("inputs.jl")
include("meshes.jl")
include("cache.jl")
include("dofs.jl")
include("spaces.jl")
include("iterative_solvers.jl")
include("pressure_operators.jl")
include("preconditioners.jl")
include("inversion.jl")
include("timesteppers.jl")
include("evolution.jl")
include("model.jl")
include("IO.jl")
include("plotting.jl")

export
x⃗,
y⃗,
z⃗,
out_dir,
set_out_dir!,
ENABLE_TIMING,
# architectures.jl
AbstractArchitecture,
CPU,
GPU,
on_architecture,
architecture,
print_memory_status,
# inputs.jl
Parameters,
SurfaceDirichletBC,
ConvectionParameterization,
EddyParameterization,
SurfaceFluxBC,
Forcings,
# meshes.jl
Mesh,
get_p_t,
compute_h_cells,
median_edge_length,
# dofs.jl
FEData,
n_free_up,
block_ranges,
# pressure_operators.jl
PressureOperators,
build_pressure_operators,
build_velocity_mass_lumped,
# preconditioners.jl
Preconditioner,
ScaledIdentity,
BlockDiagonalPreconditioner,
BlockTriangularPreconditioner,
NullspaceProjected,
FactorInverse,
DiagInverse,
KrylovInverse,
HostInverse,
ProjectedInverse,
RefreshablePreconditioner,
FunctionInverse,
InversionBlocks,
split_blocks,
augment_system,
velocity_inverse,
build_preconditioner,
schur_mass,
schur_stiffness,
schur_cahouet_chabard,
schur_geostrophic,
schur_lsc,
schur_augmented_lagrangian,
schur_exact,
# inversion.jl
InversionToolkit,
invert!,
# timesteppers.jl
BDF1,
BDF2,
# evolution.jl
EvolutionToolkit,
# model.jl
State,
Model,
set_b!,
set_state_from_file!,
evolve!,
rebuild_preconditioner!,
update_t!,
run!,
# IO.jl
save_state,
save_vtk,
save_vtk_p2,
# plotting.jl
eval_at_points,
eval_at_point,
find_H,
plot_slice,
plot_profiles,
sim_plots,
plot_sparsity_pattern

end # module