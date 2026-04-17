# define CPU and GPU architectures (see: Oceananigans.jl/src/Architectures.jl)
# the GPU architecture is extended in ext/nuPGCMCUDAExt.jl to support CUDA arrays and sparse matrices

abstract type AbstractArchitecture end
struct CPU <: AbstractArchitecture end
struct GPU <: AbstractArchitecture end

# convert types from one architecture to another
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::SparseMatrixCSC) = a

# determine architecture array is on 
architecture(::Array) = CPU()
architecture(::SparseMatrixCSC) = CPU()

# create a vector holding elements of type T on architecture 
vector_type(::CPU, T) = Vector{T}

# memory status
print_memory_status(::CPU) = println("CPU memory usage: " * string(round(Sys.maxrss() / 1e9, digits = 3)) * " GiB")