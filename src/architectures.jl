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