abstract type Preconditioner end

################################################################################

struct CgPreconditioner{M, P, W, I} <: Preconditioner
    matrix::M
    preconditioner::P
    workspace::W
    ldiv::Bool
    label::String
    itmax::I
end

function CgPreconditioner(matrix, preconditioner; ldiv=false, label="", itmax=0)
    arch = architecture(matrix)
    n = size(matrix, 1)
    T = eltype(matrix)
    VT = vector_type(arch, T)
    workspace = CgWorkspace(n, n, VT)
    workspace.x .= zero(T)
    return CgPreconditioner(matrix, preconditioner, workspace, ldiv, label, itmax)
end

function LinearAlgebra.mul!(y, cgp::CgPreconditioner, x)
    Krylov.krylov_solve!(cgp.workspace, cgp.matrix, x, cgp.workspace.x, 
                  M=cgp.preconditioner, ldiv=cgp.ldiv, itmax=cgp.itmax)#, atol=1e-6, rtol=1e-6)
    @debug begin 
        label = cgp.label
        solved = cgp.workspace.stats.solved
        niter = cgp.workspace.stats.niter 
        time = cgp.workspace.stats.timer
        @sprintf("%s iterative solve: solved=%s, niter=%d, time=%1.3e", 
                 label, solved, niter, time)
    end
    y .= cgp.workspace.x
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
    P_trial = fe_data.spaces.X_trial[2]
    P_test = fe_data.spaces.X_test[2]
    ε = params.ε
    α = params.α
    p_p = fe_data.dofs.p_p
    nu = fe_data.dofs.nu
    np = fe_data.dofs.np

    # blocks
    # A = A_inversion[1:nu, 1:nu]
    @warn "Using ν = 1 for preconditioner"
    ν = 1
    A = build_A_inversion(fe_data, params, ν; friction_only=true)
    A = A[fe_data.dofs.p_inversion, fe_data.dofs.p_inversion]
    A = A[1:nu, 1:nu]
    dropzeros!(A)
    P⁻¹ = P_block_setup(arch, A)
    P_block = Block(P⁻¹, 1:nu)

    # T⁻¹ = CG with pressure mass matrix and its diagonal
    a(p, q) = ∫( p*q )dΩ
    T = assemble_matrix(a, P_trial, P_test)[p_p, p_p]/(α^2*ε^2)
    T_prec = Diagonal(on_architecture(arch, Vector(1 ./ diag(T))))
    T = on_architecture(arch, T)
    T⁻¹ = CgPreconditioner(T, T_prec; ldiv=false, label="T-block", itmax=0)
    T_block = Block(T⁻¹, nu+1:nu+np)

    return BlockDiagonalPreconditioner([P_block, T_block])
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

    P_prec = KrylovPreconditioners.kp_ilu0(P)
    return CgPreconditioner(P, P_prec; ldiv=true, label="P$tag-block", itmax=100)

    # h = 2.56e-2
    # dim = 3
    # @warn "Assuming h = $h and dim = $dim"
    # P_prec = Diagonal(on_architecture(GPU(), 1/h^dim*ones(size(A, 1))))

    # # P_prec = Diagonal(on_architecture(GPU(), Vector(1 ./ diag(A))))
    # return CgPreconditioner(P, P_prec; ldiv=false, label="P$tag-block", itmax=1000)
end

function LinearAlgebra.mul!(y, bdp::BlockDiagonalPreconditioner, x)
    for block in bdp.blocks
        xb = @view x[block.indices]
        yb = @view y[block.indices]
        mul!(yb, block.P⁻¹, xb)
    end
    return y
end
