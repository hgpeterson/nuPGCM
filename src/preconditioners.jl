abstract type Preconditioner end

################################################################################

struct LUPreconditioner{M} <: Preconditioner
    M::M
end

function LUPreconditioner(::CPU, A)
    return LUPreconditioner(lu(A))
end

function LUPreconditioner(::GPU, A)
    A_gpu = on_architecture(GPU(), A)
    return LUPreconditioner(kp_ilu0(A_gpu))
end

function LinearAlgebra.mul!(y, lup::LUPreconditioner, x)
    ldiv!(y, lup.M, x)
    return y
end

################################################################################

struct CgPreconditioner{M, P, S} <: Preconditioner
    matrix::M
    preconditioner::P
    solver::S
end

function CgPreconditioner(matrix, preconditioner)
    arch = architecture(matrix)
    n = size(matrix, 1)
    T = eltype(matrix)
    VT = vector_type(arch, T)
    solver = CgSolver(n, n, VT)
    solver.x .= zero(T)
    return CgPreconditioner(matrix, preconditioner, solver)
end

function LinearAlgebra.mul!(y, cgp::CgPreconditioner, x)
    Krylov.solve!(cgp.solver, cgp.matrix, x, M=cgp.preconditioner, ldiv=false,
                  atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=0)
    @debug begin 
        solved = cgp.solver.stats.solved
        niter = cgp.solver.stats.niter 
        time = cgp.solver.stats.timer
        "CgPreconditioner iterative solve: solved=$solved, niter=$niter, time=$time" 
    end
    y .= cgp.solver.x
    return y
end

################################################################################

"""
From Elman2014:

Stokes problem is 
    [ A  B^T ] [ u ] = [ f ]
    [ B  0   ] [ p ]   [ g ]
    
Preconditioner of the form
    M = [ P  0 ]
        [ 0  T ]
where P ∼ A and T ∼ B A^-1 B^T
"""
struct BlockDiagonalPreconditioner{P, T, I} <: Preconditioner
    P⁻¹::P
    T⁻¹::T
    n::I
    m::I
end

function BlockDiagonalPreconditioner(arch::AbstractArchitecture, params::Parameters, fe_data::FEData, A_inversion)
    # unpack
    dΩ = fe_data.mesh.dΩ
    P = fe_data.spaces.X_trial[4]
    Q = fe_data.spaces.X_test[4]
    ε = params.ε
    α = params.α
    p_p = fe_data.dofs.p_p
    N = fe_data.dofs.nu + fe_data.dofs.nv + fe_data.dofs.nw
    np = fe_data.dofs.np

    # blocks
    dropzeros!(A_inversion)
    A = A_inversion[1:N, 1:N]
    # B = A_inversion[N+1:end, 1:N]
    # BT = A_inversion[1:N, N+1:end]

    # P⁻¹ = LU-factorization of A
    P⁻¹ = LUPreconditioner(arch, A)

    # T⁻¹ = CG with pressure mass matrix
    a(p, q) = ∫( p*q )dΩ
    M = assemble_matrix(a, P, Q) 
    M = M[p_p, p_p]/α^2/ε^2
    P_M = Diagonal(on_architecture(arch, Vector(1 ./ diag(M))))
    M = on_architecture(arch, M)
    T⁻¹ = CgPreconditioner(M, P_M)

    return BlockDiagonalPreconditioner(P⁻¹, T⁻¹, N, np-1)
end

function LinearAlgebra.mul!(y, bdp::BlockDiagonalPreconditioner, x)
    y1 = @view y[1:bdp.n]
    y2 = @view y[bdp.n+1:end]
    x1 = @view x[1:bdp.n]
    x2 = @view x[bdp.n+1:end]
    mul!(y1, bdp.P⁻¹, x1)
    mul!(y2, bdp.T⁻¹, x2)
    return y
end
