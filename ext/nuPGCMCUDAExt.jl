module nuPGCMCUDAExt

using nuPGCM
using CUDA
using CUDA.cuSPARSE
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

# `gpu_perm` holds the CSC → CSR permutation of the nonzeros and a CPU buffer for
# the permuted values.
function nuPGCM.update_A!(solver_A::CuSparseMatrixCSR, A_new::SparseMatrixCSC, gpu_perm::Ref)
    if gpu_perm[] === nothing
        gpu_perm[] = (_csc_to_csr_perm(A_new, solver_A), similar(A_new.nzval))
    end
    perm, staging = gpu_perm[]::Tuple{Vector{Int}, Vector{Float64}}
    staging .= @view A_new.nzval[perm]
    copyto!(solver_A.nzVal, staging)
    return solver_A
end

"""
    perm = _csc_to_csr_perm(A_csc, A_csr)

Permutation such that `A_csc.nzval[perm] == Array(A_csr.nzVal)` for two matrices
with the same sparsity pattern.
"""
function _csc_to_csr_perm(A_csc::SparseMatrixCSC, A_csr::CuSparseMatrixCSR)
    rowPtr = Array(A_csr.rowPtr)
    colVal = Array(A_csr.colVal)
    n = size(A_csc, 1)
    perm = Vector{Int}(undef, nnz(A_csc))
    t = 0
    for row in 1:n
        for idx in rowPtr[row]:(rowPtr[row+1]-1)
            col = colVal[idx]
            rng = A_csc.colptr[col]:(A_csc.colptr[col+1]-1)
            t += 1
            perm[t] = searchsortedfirst(A_csc.rowval, row, first(rng), last(rng), Base.Order.Forward)
        end
    end
    return perm
end

end # module