module nuPGCM

using Gridap
using GridapGmsh
using Gmsh: gmsh
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using CuthillMcKee
using SparseArrays
using PyPlot
using HDF5
using Printf
import Base.string

# define CPU and GPU architectures (credit: Oceananigans.jl)
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end
struct GPU <: AbstractArchitecture end

# convert types from one architecture to another (credit: Oceananigans.jl)
on_architecture(::CPU, a::Array) = a
on_architecture(::GPU, a::Array) = CuArray(a)

on_architecture(::CPU, a::CuArray) = Array(a)
on_architecture(::GPU, a::CuArray) = a

on_architecture(::CPU, a::SparseMatrixCSC) = a
on_architecture(::GPU, a::SparseMatrixCSC) = CuSparseMatrixCSR(a)

on_architecture(::CPU, a::CuSparseMatrixCSR) = SparseMatrixCSC(a)
on_architecture(::GPU, a::CuSparseMatrixCSR) = a

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
include("meshes.jl")
include("plotting.jl")
include("IO.jl")
include("spaces.jl")
include("matrices.jl")

export 
AbstractArchitecture,
CPU,
GPU,
on_architecture,
AbstractDimension,
TwoD,
ThreeD,
# utils.jl
chebyshev_nodes,
chebyshev_grid,
hrs_mins_secs,
sci_notation,
trapz,
cumtrapz,
nan_max,
nan_min,
# meshes.jl
Mesh,
get_p_t,
get_p_to_t,
# plotting.jl
nan_eval,
plot_slice,
plot_profiles,
sim_plots,
plot_sparsity_pattern,
# IO.jl
write_sparse_matrix,
read_sparse_matrix,
save_state,
save_state_vtu,
load_state,
# spaces.jl
setup_FESpaces,
unpack_spaces,
# matrices.jl
∂x,
∂y,
∂z,
assemble_LHS_inversion,
assemble_RHS_inversion,
assemble_LHS_adv_diff,
assemble_RHS_diff

end # module