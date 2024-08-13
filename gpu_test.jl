using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
import LinearAlgebra: ldiv!

# Optional -- Compute a permutation vector p such that A[:,p] has no zero diagonal
p = zfd(A_cpu)
p .+= 1
A_cpu = A_cpu[:,p]

# Transfer the linear system from the CPU to the GPU
A_gpu = CuSparseMatrixCSR(A_cpu)  # A_gpu = CuSparseMatrixCSC(A_cpu)
b_cpu = rand(eltype(A_cpu), size(A_cpu, 1))
b_gpu = CuVector(b_cpu)

# LUPreconditioner
struct LUPreconditioner{T}
    L::T
    U::T
end
F = ilu(A_cpu, τ=1e-4)
L_gpu = CuSparseMatrixCSR(F.L)
U_gpu = CuSparseMatrixCSR(F.U')
F = LUPreconditioner(L_gpu, U_gpu)

# solve LUx = y
z = CUDA.zeros(eltype(b_gpu), length(b_gpu))
function ldiv!(x, F::LUPreconditioner, y)
    ldiv!(z, LowerTriangular(F.L), y)
    ldiv!(x, UpperTriangular(F.U), z)
    return x
end

# Solve a non-Hermitian system with an ILU(0) preconditioner on GPU
x̄, stats = bicgstab(A_gpu, b_gpu, M=F, verbose=1, ldiv=true)
# x̄, stats = gmres(A_cpu, b_cpu, M=F, verbose=1, ldiv=true)
# x̄, stats = gmres(A_gpu, b_gpu, verbose=1)

# Recover the solution of Ax = b with the solution of A[:,p]x̄ = b
invp = invperm(p)
x = Array(x̄[invp])

# true solution
x̄_true = A_cpu \ b_cpu
x_true = x̄_true[invp]
println("Absolute error: ", norm(x - x_true))
println("Relative error: ", norm(x - x_true) / norm(x_true))
