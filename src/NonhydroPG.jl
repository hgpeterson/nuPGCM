module NonhydroPG

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

# vector type for iterative solvers
vector_type(::CPU) = Vector{Float64} 
vector_type(::GPU) = CuVector{Float64}

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

# output folder
global out_folder = ""
function set_out_folder!(folder::AbstractString)
    global out_folder = folder
    if !isdir(out_folder)
        println("creating folder: ", out_folder)
        mkdir(out_folder)
    end
    if !isdir("$out_folder/images")
        println("creating subfolder: ", out_folder, "/images")
        mkdir("$out_folder/images")
    end
    if !isdir("$out_folder/data")
        println("creating subfolder: ", out_folder, "/data")
        mkdir("$out_folder/data")
    end
    flush(stdout)
    flush(stderr)
end

include("model.jl")
include("utils.jl")
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
out_folder,
set_out_folder!,
# utils.jl
chebyshev_nodes,
hrs_mins_secs,
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
assemble_LHS_inversion,
assemble_RHS_inversion,
assemble_LHS_evolution

end # module