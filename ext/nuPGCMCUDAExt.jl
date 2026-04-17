module nuPGCMCUDAExt

using nuPGCM
using CUDA
using CUDA.CUSPARSE
using SparseArrays

function __init__()
    if CUDA.functional()
        msg = "CUDA device(s):\n"
        for (gpu, dev) in enumerate(CUDA.devices())
            msg *= "$dev: $(CUDA.name(dev))\n"
        end
        @info msg
    end
end

export on_architecture, 
architecture, 
vector_type,
print_memory_status

# implement `on_architecture`, `architecture`, and `vector_type` for CUDA
nuPGCM.on_architecture(::GPU, a::Array) = CuArray(a)
nuPGCM.on_architecture(::CPU, a::CuArray) = Array(a)
nuPGCM.on_architecture(::GPU, a::CuArray) = a
nuPGCM.on_architecture(::GPU, a::SparseMatrixCSC) = CuSparseMatrixCSR(a)
nuPGCM.on_architecture(::CPU, a::CuSparseMatrixCSR) = SparseMatrixCSC(a)
nuPGCM.on_architecture(::GPU, a::CuSparseMatrixCSR) = a
nuPGCM.architecture(::CuArray) = GPU()
nuPGCM.architecture(::CuSparseMatrixCSR) = GPU()
nuPGCM.vector_type(::GPU, T) = CuVector{T}
nuPGCM.print_memory_status(::GPU) = CUDA.pool_status()

end # module