"""
    A, rhs = getMatrices()   

Compute matrices for 1D equations.
"""
function getMatrices()
    nVars = 3
    nPts = nVars*nz̃

    umap = reshape(1:nPts, nVars, nz̃)    
    A = Tuple{Int64,Int64,Float64}[] # LHS matrix 
    rhs = zeros(nPts)                # RHS vector

    # Main loop, insert stencil in matrices for each node point
    for j=2:nz̃-1
        # dz̃ stencil
        fd_z̃ = mkfdstencil(z̃[j-1:j+1], z̃[j], 1)
        ν_z̃ = sum(fd_z̃.*ν[j-1:j+1])
        κ_z̃ = sum(fd_z̃.*κ[j-1:j+1])

        # dz̃z̃ stencil
        fd_z̃z̃ = mkfdstencil(z̃[j-1:j+1], z̃[j], 2)

        # eqtn 1: -ṽ - S*b̃ - dz̃(ν*dz̃(ũ)) = 0 
        row = umap[1, j]
        push!(A, (row, umap[2, j],   -1.0))
        push!(A, (row, umap[1, j-1], -(ν_z̃*fd_z̃[1] + ν[j]*fd_z̃z̃[1])))
        push!(A, (row, umap[1, j],   -(ν_z̃*fd_z̃[2] + ν[j]*fd_z̃z̃[2])))
        push!(A, (row, umap[1, j+1], -(ν_z̃*fd_z̃[3] + ν[j]*fd_z̃z̃[3])))

        # eqtn 2: ũ - dz̃(ν*dz̃(ṽ)) = 0 
        row = umap[2, j]
        push!(A, (row, umap[1, j],   1.0))
        push!(A, (row, umap[2, j-1], -(ν_z̃*fd_z̃[1] + ν[j]*fd_z̃z̃[1])))
        push!(A, (row, umap[2, j],   -(ν_z̃*fd_z̃[2] + ν[j]*fd_z̃z̃[2])))
        push!(A, (row, umap[2, j+1], -(ν_z̃*fd_z̃[3] + ν[j]*fd_z̃z̃[3])))

        # eqtn 3: ũ - dz̃(κ*dz̃(b̃)) = dz̃(κ) 
        row = umap[3, j]
        push!(A, (row, umap[1, j],   1.0))
        push!(A, (row, umap[3, j-1], -(κ_z̃*fd_z̃[1] + κ[j]*fd_z̃z̃[1])))
        push!(A, (row, umap[3, j],   -(κ_z̃*fd_z̃[2] + κ[j]*fd_z̃z̃[2])))
        push!(A, (row, umap[3, j+1], -(κ_z̃*fd_z̃[3] + κ[j]*fd_z̃z̃[3])))
        rhs[row] = κ_z̃
    end

    # Boundary Conditions: Bottom
    # ũ = 0
    row = umap[1, 1] 
    push!(A, (row, row, 1.0))
    # ṽ = 0
    row = umap[2, 1] 
    push!(A, (row, row, 1.0))
    # dz̃(b) = -1
    row = umap[3, 1] 
    fd_z̃ = mkfdstencil(z̃[1:3], z̃[1], 1)
    push!(A, (row, umap[3, 1], fd_z̃[1]))
    push!(A, (row, umap[3, 2], fd_z̃[2]))
    push!(A, (row, umap[3, 3], fd_z̃[3]))

    # Boundary Conditions: Top
    fd_z̃ = mkfdstencil(z̃[nz̃-2:nz̃], z̃[nz̃], 1)
    # dz̃(ũ) = 0
    row = umap[1, nz̃] 
    push!(A, (row, umap[1, nz̃-2], fd_z̃[1]))
    push!(A, (row, umap[1, nz̃-1], fd_z̃[2]))
    push!(A, (row, umap[1, nz̃],   fd_z̃[3]))
    # dz̃(ṽ) = 0
    row = umap[2, nz̃] 
    push!(A, (row, umap[2, nz̃-2], fd_z̃[1]))
    push!(A, (row, umap[2, nz̃-1], fd_z̃[2]))
    push!(A, (row, umap[2, nz̃],   fd_z̃[3]))
    # dz̃(b) = 0
    row = umap[3, nz̃]
    push!(A, (row, umap[3, nz̃-2], fd_z̃[1]))
    push!(A, (row, umap[3, nz̃-1], fd_z̃[2]))
    push!(A, (row, umap[3, nz̃],   fd_z̃[3]))

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nPts, nPts)

    return A, rhs
end

"""
    b = steadyState()

Compute canonical steady state.
"""
function steadyState()
    # grid points
    nVars = 3
    nPts = nVars*nz̃

    # for flattening for matrix mult
    umap = reshape(1:nPts, nVars, nz̃)

    # get matrices and vectors
    A, rhs = getMatrices()

    # boundaries
    rhs[umap[1, 1]]  = 0  # u = 0 bot
    rhs[umap[1, nz̃]] = 0  # u decay top
    rhs[umap[2, 1]]  = 0  # v = 0 bot
    rhs[umap[2, nz̃]] = 0  # v decay top
    rhs[umap[3, 1]]  = -1 # b flux bot
    rhs[umap[3, nz̃]] = 0  # b flux top

    # solve
    solVec = A\rhs

    # gather solution and rotate
    sol = reshape(solVec, 3, nz̃)
    ũ = sol[1, :]
    ṽ = sol[2, :]
    b̃ = sol[3, :]

    # save data
    saveCheckpointRot(b̃, ũ, ṽ, -1, -42, 999)

    return b
end
