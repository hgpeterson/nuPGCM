module nuPGCM

using Gridap
using GridapGmsh
using Gmsh: gmsh
using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using CuthillMcKee
using JLD2
using LinearAlgebra
using SparseArrays
using Krylov
using PyPlot
using HDF5
using Printf

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
    global out_dir = dir
    @info "Output directory set to '$dir'"

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

# include all the module code
include("architectures.jl")
include("utils.jl")
include("parameters.jl")
include("spaces.jl")
include("dofs.jl")
include("meshes.jl")
include("matrices.jl")
include("preconditioners.jl")
include("inversion.jl")
include("evolution.jl")
include("model.jl")
include("IO.jl")
include("plotting.jl")

export 
out_dir,
set_out_dir!,
# architectures.jl
AbstractArchitecture,
CPU,
GPU,
on_architecture,
architecture,
# utils.jl
chebyshev_nodes,
hrs_mins_secs,
sci_notation,
trapz,
cumtrapz,
nan_max,
nan_min,
# parameters.jl
Parameters,
# spaces.jl
Spaces,
# dofs.jl
get_n_dof,
# meshes.jl
Mesh,
get_p_t,
get_p_to_t,
# matrices.jl
∂x,
∂y,
∂z,
build_matrices,
build_inversion_matrices,
build_evolution_matrices,
# preconditioners.jl
mul!,
# inversion.jl
InversionToolkit,
invert!,
# evolution.jl
EvolutionToolkit,
# model.jl
State,
Model,
set_b!,
# IO.jl
save_state,
load_state_from_file!,
# plotting.jl
nan_eval,
plot_slice,
plot_profiles,
sim_plots,
plot_sparsity_pattern

end # module