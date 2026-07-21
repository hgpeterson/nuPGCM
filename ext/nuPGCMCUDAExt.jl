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

"""
    update_A!(solver_A::CuSparseMatrixCSR, A_new::SparseMatrixCSC, gpu_perm)

In-place refresh of a GPU CSR matrix from a same-pattern CPU CSC matrix.
`CuSparseMatrixCSR(::SparseMatrixCSC)` stores nonzeros in row-major (CSR)
order, not `SparseMatrixCSC`'s column-major order, so `A_new.nzval` must be
permuted before copying. The permutation depends only on the (fixed)
sparsity pattern, so it is computed once — from `solver_A` itself, which was
already correctly built via `CuSparseMatrixCSR(A_new)` at construction time —
and cached in `gpu_perm` for every subsequent call. The permuted values are
staged into a plain `Vector` first (`gpu_perm[]` also holds this buffer) so
the host→device transfer is a single bulk `copyto!`, not fancy-indexed scalar
GPU writes (which `CUDA.jl` disallows and would otherwise error on).
"""
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

Permutation such that `A_csc.nzval[perm] == Array(A_csr.nzVal)` for any
`A_csc`/`A_csr` pair sharing the same sparsity pattern (only values may
differ from the pair used to compute `perm`). Derived by locating, for each
CSR entry's `(row, col)`, its linear index in the CSC structure.
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
            pos = searchsortedfirst(A_csc.rowval, row, first(rng), last(rng), Base.Order.Forward)
            @assert pos <= last(rng) && A_csc.rowval[pos] == row "CSR/CSC pattern mismatch at ($row,$col)"
            t += 1
            perm[t] = pos
        end
    end
    return perm
end

end # module