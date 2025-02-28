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

# define CPU and GPU architectures (see: Oceananigans.jl/src/Architectures.jl)
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end
struct GPU <: AbstractArchitecture end

# convert types from one architecture to another
on_architecture(::CPU, a::Array) = a
on_architecture(::GPU, a::Array) = CuArray(a)

on_architecture(::CPU, a::CuArray) = Array(a)
on_architecture(::GPU, a::CuArray) = a

on_architecture(::CPU, a::SparseMatrixCSC) = a
on_architecture(::GPU, a::SparseMatrixCSC) = CuSparseMatrixCSR(a)

on_architecture(::CPU, a::CuSparseMatrixCSR) = SparseMatrixCSC(a)
on_architecture(::GPU, a::CuSparseMatrixCSR) = a

# determine architecture array is on 
architecture(::Array) = CPU()
architecture(::CuArray) = GPU()
architecture(::SparseMatrixCSC) = CPU()
architecture(::CuSparseMatrixCSR) = GPU()

# physical dimension of the problem
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

include("utils.jl")
include("plotting.jl")
include("IO.jl")
include("spaces.jl")
include("meshes.jl")
include("matrices.jl")
include("inversion.jl")
include("preconditioners.jl")

export 
out_dir,
set_out_dir!,
AbstractArchitecture,
CPU,
GPU,
on_architecture,
architecture,
AbstractDimension,
TwoD,
ThreeD,
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
# IO.jl
save_state,
save_state_vtu,
load_state,
# spaces.jl
unpack_spaces,
# meshes.jl
Mesh,
get_n_dof,
get_p_t,
get_p_to_t,
# matrices.jl
∂x,
∂y,
∂z,
build_matrices,
# inversion.jl
InversionToolkit,
invert!,
# preconditioners.jl
mul!

end # module