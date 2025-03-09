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
import Base.string

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

#TODO: Remove
abstract type AbstractDimension end
struct TwoD <: AbstractDimension 
    n::Int
end
struct ThreeD <: AbstractDimension 
    n::Int
end
TwoD() = TwoD(2)
ThreeD() = ThreeD(3)
string(::TwoD) = "2D"
string(::ThreeD) = "3D"

# include all the module code
include("architectures.jl")
include("utils.jl")
include("plotting.jl")
include("parameters.jl")
include("spaces.jl")
include("dofs.jl")
include("meshes.jl")
include("matrices.jl")
include("preconditioners.jl")
include("inversion.jl")
include("evolution.jl")
include("state.jl")
include("model.jl")

export 
out_dir,
set_out_dir!,
AbstractDimension,
TwoD,
ThreeD,
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
# plotting.jl
nan_eval,
plot_slice,
plot_profiles,
sim_plots,
plot_sparsity_pattern,
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
# state.jl
State,
rest_state,
set_state!,
save,
load_state,
# model.jl
Model

end # module