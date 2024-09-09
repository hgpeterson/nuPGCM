module NonhydroPG

using Gridap
using GridapGmsh
using Gmsh: gmsh
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using PyPlot
using SparseArrays
using HDF5
using Printf

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

include("utils.jl")
include("meshes.jl")
include("plotting.jl")
include("IO.jl")
include("matrices.jl")

export 
chebyshev_nodes,
hrs_mins_secs,
AbstractArchitecture,
CPU,
GPU,
on_architecture,
Mesh,
get_p_t,
get_p_to_t,
nan_eval,
unpack_fefunction,
quick_plot,
plot_yslice,
plot_profiles,
plot_sparsity_pattern,
plot_u_sfc,
write_sparse_matrix,
read_sparse_matrix,
save_state,
load_state,
assemble_LHS_inversion,
assemble_RHS_inversion,
assemble_LHS_evolution

end # module