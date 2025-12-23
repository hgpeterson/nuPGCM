abstract type Preconditioner end

################################################################################

struct CgPreconditioner{M, P, S} <: Preconditioner
    matrix::M
    preconditioner::P
    solver::S
    ldiv::Bool
    label::String
end

function CgPreconditioner(matrix, preconditioner; ldiv=false, label="")
    arch = architecture(matrix)
    n = size(matrix, 1)
    T = eltype(matrix)
    VT = vector_type(arch, T)
    solver = CgSolver(n, n, VT)
    solver.x .= zero(T)
    return CgPreconditioner(matrix, preconditioner, solver, ldiv, label)
end

function LinearAlgebra.mul!(y, cgp::CgPreconditioner, x)
    Krylov.solve!(cgp.solver, cgp.matrix, x, M=cgp.preconditioner, ldiv=cgp.ldiv,
                  atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=0)
    @debug begin 
        solved = cgp.solver.stats.solved
        niter = cgp.solver.stats.niter 
        time = cgp.solver.stats.timer
        "$(cgp.label) iterative solve: solved=$solved, niter=$niter, time=$time" 
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
    P_trial = fe_data.spaces.X_trial[4]
    P_test = fe_data.spaces.X_test[4]
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

    # # P⁻¹ = CG with A and its LU-factorization
    # P = A
    # P_prec = lu(A)
    # P⁻¹ = CgPreconditioner(P, P_prec; ldiv=true, label="P-block")

    # P⁻¹ = CG with A and its Incomplete LU-factorization
    P = on_architecture(arch, A)
    P_prec = KrylovPreconditioners.kp_ilu0(P)
    P⁻¹ = CgPreconditioner(P, P_prec; ldiv=true, label="P-block")

    # # P⁻¹ = CG with A and its diagonal
    # P = A
    # P_prec = Diagonal(on_architecture(arch, Vector(1 ./ diag(P))))
    # P = on_architecture(arch, P)
    # P⁻¹ = CgPreconditioner(P, P_prec; ldiv=false, label="P-block")

    # T⁻¹ = CG with pressure mass matrix and its diagonal
    a(p, q) = ∫( p*q )dΩ
    T = assemble_matrix(a, P_trial, P_test)[p_p, p_p]/(α^2*ε^2)
    T_prec = Diagonal(on_architecture(arch, Vector(1 ./ diag(T))))
    T = on_architecture(arch, T)
    T⁻¹ = CgPreconditioner(T, T_prec; ldiv=false, label="T-block")

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
