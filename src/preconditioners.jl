abstract type Preconditioner end

################################################################################

struct CgPreconditioner{M, P, S, I} <: Preconditioner
    matrix::M
    preconditioner::P
    solver::S
    ldiv::Bool
    label::String
    itmax::I
end

function CgPreconditioner(matrix, preconditioner; ldiv=false, label="", itmax=0)
    arch = architecture(matrix)
    n = size(matrix, 1)
    T = eltype(matrix)
    VT = vector_type(arch, T)
    solver = CgSolver(n, n, VT)
    solver.x .= zero(T)
    return CgPreconditioner(matrix, preconditioner, solver, ldiv, label, itmax)
end

function LinearAlgebra.mul!(y, cgp::CgPreconditioner, x)
    Krylov.solve!(cgp.solver, cgp.matrix, x, cgp.solver.x, 
                  M=cgp.preconditioner, ldiv=cgp.ldiv, itmax=cgp.itmax)#, atol=1e-6, rtol=1e-6)
    @debug begin 
        label = cgp.label
        solved = cgp.solver.stats.solved
        niter = cgp.solver.stats.niter 
        time = cgp.solver.stats.timer
        @sprintf("%s iterative solve: solved=%s, niter=%d, time=%1.3e", 
                 label, solved, niter, time)
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
struct BlockDiagonalPreconditioner{B} <: Preconditioner
    blocks::B
end

struct Block{P, I}
    P⁻¹::P
    indices::I
end

function BlockDiagonalPreconditioner(arch::AbstractArchitecture, params::Parameters, fe_data::FEData, A_inversion)
    # unpack
    dΩ = fe_data.mesh.dΩ
    P_trial = fe_data.spaces.X_trial[4]
    P_test = fe_data.spaces.X_test[4]
    ε = params.ε
    α = params.α
    p_p = fe_data.dofs.p_p
    nu = fe_data.dofs.nu
    nv = fe_data.dofs.nv
    nw = fe_data.dofs.nw
    N = nu + nv + nw
    np = fe_data.dofs.np

    # blocks
    A = A_inversion[1:N, 1:N]
    nu = fe_data.dofs.nu
    nv = fe_data.dofs.nv
    A[1:nu, nu+1:nu+nv] .= 0  # remove Coriolis terms
    A[nu+1:nu+nv, 1:nu] .= 0  # remove Coriolis terms
    dropzeros!(A)
    P⁻¹ = P_block_setup(arch, A)
    P_block = Block(P⁻¹, 1:N)

    # indices_u = 1:nu
    # indices_v = nu+1:nu+nv
    # indices_w = nu+nv+1:N
    # dropzeros!(A_inversion)
    # Au = A_inversion[indices_u, indices_u]
    # Av = A_inversion[indices_v, indices_v]
    # Aw = A_inversion[indices_w, indices_w]

    # Pu⁻¹ = P_block_setup(arch, Au, tag="u")
    # Pv⁻¹ = P_block_setup(arch, Av, tag="v")
    # Pw⁻¹ = P_block_setup(arch, Aw, tag="w")
    # Pu_block = Block(Pu⁻¹, indices_u)
    # Pv_block = Block(Pv⁻¹, indices_v)
    # Pw_block = Block(Pw⁻¹, indices_w)

    # T⁻¹ = CG with pressure mass matrix and its diagonal
    a(p, q) = ∫( p*q )dΩ
    T = assemble_matrix(a, P_trial, P_test)[p_p, p_p]/(α^2*ε^2)
    T_prec = Diagonal(on_architecture(arch, Vector(1 ./ diag(T))))
    T = on_architecture(arch, T)
    T⁻¹ = CgPreconditioner(T, T_prec; ldiv=false, label="T-block", itmax=0)
    T_block = Block(T⁻¹, N+1:N+np)

    return BlockDiagonalPreconditioner([P_block, T_block])
    # return BlockDiagonalPreconditioner([Pu_block, Pv_block, Pw_block, T_block])
end

function P_block_setup(::CPU, A; tag="")
    # P⁻¹ = CG with A and its LU-factorization
    P = A
    P_prec = lu(P)
    return CgPreconditioner(P, P_prec; ldiv=true, label="P$tag-block")
end

function P_block_setup(::GPU, A; tag="")
    # P⁻¹ = CG with A and its Incomplete LU-factorization
    P = on_architecture(GPU(), A)

    # P_prec = KrylovPreconditioners.kp_ilu0(P)
    # return CgPreconditioner(P, P_prec; ldiv=true, label="P$tag-block", itmax=0)

    h = 0.028
    dim = 3
    P_prec = Diagonal(on_architecture(GPU(), 1/h^dim*ones(size(A, 1))))

    # P_prec = Diagonal(on_architecture(GPU(), Vector(1 ./ diag(A))))
    return CgPreconditioner(P, P_prec; ldiv=false, label="P$tag-block", itmax=0)
end

function LinearAlgebra.mul!(y, bdp::BlockDiagonalPreconditioner, x)
    for block in bdp.blocks
        xb = @view x[block.indices]
        yb = @view y[block.indices]
        mul!(yb, block.P⁻¹, xb)
    end
    return y
end
