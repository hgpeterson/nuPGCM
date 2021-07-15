"""
    A, diffVec = getRHS()   

If sol = (ũ, ṽ, b̃, P_x) then we compute `A` and `diffVec` such that sol_t = A*sol + diffVec
and `A` also contains the proper boundary conditions.
"""
function getMatrices()
    nVars = 3
    nPts = nVars*nz̃ + 1

    umap = reshape(1:(nPts-1), nVars, nz̃)    
    A = Tuple{Int64,Int64,Float64}[]  
    diffVec = zeros(nPts)

    # Main loop, insert stencil in matrices for each node point
    for j=2:nz̃-1
        # dz̃ stencil
        fd_z̃ = mkfdstencil(z̃[j-1:j+1], z̃[j], 1)
        ν_z̃ = sum(fd_z̃.*ν[j-1:j+1])
        κ_z̃ = sum(fd_z̃.*κ[j-1:j+1])

        # dz̃z̃ stencil
        fd_z̃z̃ = mkfdstencil(z̃[j-1:j+1], z̃[j], 2)

        # 1st eqtn: ũ_t̃ = ṽ - P_x + S*b̃ + Pr*(κ*ũ_z̃)_z̃
        row = umap[1, j]
        # first term
        push!(A, (row, umap[2, j], 1.0))
        # second term
        push!(A, (row, nPts, -1.0))
        # third term
        push!(A, (row, umap[3, j], S))
        # fourth term: dz̃(ν*dz̃(ũ))) = dz̃(ν)*dz̃(ũ) + ν*dz̃z̃(ũ)
        push!(A, (row, umap[1, j-1], ν_z̃*fd_z̃[1] + ν[j]*fd_z̃z̃[1]))
        push!(A, (row, umap[1, j],   ν_z̃*fd_z̃[2] + ν[j]*fd_z̃z̃[2]))
        push!(A, (row, umap[1, j+1], ν_z̃*fd_z̃[3] + ν[j]*fd_z̃z̃[3]))

        # 2nd eqtn: ṽ_t̃ = -ũ + (ν*ṽ_z̃)_z̃
        row = umap[2, j]
        # first term:
        push!(A, (row, umap[1, j], -1.0))
        # second term: dz̃(ν*dz̃(ṽ))) = dz̃(ν)*dz̃(ṽ) + ν*dz̃z̃(ṽ)
        push!(A, (row, umap[2, j-1], ν_z̃*fd_z̃[1] + ν[j]*fd_z̃z̃[1]))
        push!(A, (row, umap[2, j],   ν_z̃*fd_z̃[2] + ν[j]*fd_z̃z̃[2]))
        push!(A, (row, umap[2, j+1], ν_z̃*fd_z̃[3] + ν[j]*fd_z̃z̃[3]))

        # 3rd eqtn: b̃_t̃ = -ũ + [κ*(1 + b̃_z̃)]_z̃
        row = umap[3, j]
        # first term
        push!(A, (row, umap[1, j], -1.0))
        # second term: dz̃(κ(N^2 + dz̃(b̃))) = N^2*dz̃(κ) + dz̃(κ)*dz̃(b̃) + κ*dz̃z̃(b̃)
        push!(A, (row, umap[3, j-1], (κ_z̃*fd_z̃[1] + κ[j]*fd_z̃z̃[1])))
        push!(A, (row, umap[3, j],   (κ_z̃*fd_z̃[2] + κ[j]*fd_z̃z̃[2])))
        push!(A, (row, umap[3, j+1], (κ_z̃*fd_z̃[3] + κ[j]*fd_z̃z̃[3])))
        diffVec[row] = N^2*κ_z̃
    end

    # Boundary Conditions: Bottom
    # ũ = 0
    row = umap[1, 1] 
    push!(A, (row, row, 1.0))
    # ṽ = 0
    row = umap[2, 1] 
    push!(A, (row, row, 1.0))
    # b̃_z̃ = -N^2
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
    # dz̃(b̃) = 0
    row = umap[3, nz̃]
    push!(A, (row, umap[3, nz̃-2], fd_z̃[1]))
    push!(A, (row, umap[3, nz̃-1], fd_z̃[2]))
    push!(A, (row, umap[3, nz̃],   fd_z̃[3]))

    # transport constraint
    row = nPts
    if canonical == true
        # canonical 1D: P_x = constant in time
        push!(A, (row, nPts, 1.0))
    else
        # transport-constrained 1D: P_x such that ∫ ũ dz̃ = 0
        for j=1:nz̃-1
            # trapez̃oidal rule: (ũ_j+1 + ũ_j)/2 * Δz̃_j
            push!(A, (row, umap[1, j],   (z̃[j+1] - z̃[j])/2))
            push!(A, (row, umap[1, j+1], (z̃[j+1] - z̃[j])/2))
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nPts, nPts)

    return A, diffVec
end

"""
    LHS, RHS = getLHSandRHS(Δt̃, A, α, bottomBdy, topBdy)

Get left- and right-hand-side matrices for time stepping where
    (x^(n+1) - x^n)/Δt̃ = α*A*x^n + (1 - α)*A*x^(n+1) + y
So that 
    LHS = I/Δt̃ - (1 - α)*A,
    RHS = I/Δt̃ + α*A.
"""
function getLHSandRHS(Δt̃, A, α, bottomBdy, topBdy)
    nVars = 3
    nPts = nVars*nz̃ + 1

    # (x^(n+1) - x^n)/Δt̃ = α*A*x^n + (1 - α)*A*x^(n+1)
    LHS = I/Δt̃ - (1 - α)*A 
    RHS = I/Δt̃ + α*A 

    # boundary conditions and integral constraint
    LHS[bottomBdy, :] = A[bottomBdy, :]
    LHS[topBdy, :] = A[topBdy, :]
    LHS[nPts, :] = A[nPts, :]

    return LHS, RHS
end

"""
    sol = evolve(tFinal)

Solve 1D equations for `tFinal` seconds.
"""
function evolve(tFinal)
    # grid points
    nVars = 3
    nPts = nVars*nz̃ + 1

    # timestep
    nSteps = Int64(ceil(tFinal/Δt̃))
    if tSave === nothing
        nStepsSave = nothing
    else
        nStepsSave = Int64(floor(tSave/Δt̃))
    end

    # for flattening for matrix mult
    umap = reshape(1:(nPts-1), nVars, nz̃)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nz̃]

    # get matrices and vectors
    A, diffVec = getMatrices()

    # left- and right-hand-side matrices for evolution
    LHS, RHS = getLHSandRHS(Δt̃, A, α, bottomBdy, topBdy)
    LHS = lu(LHS)

    # initial condition
    t̃ = 0
    sol = zeros(nPts)
    # start with far-field geostrophic flow
    sol[umap[2, :]] .= ṽ_0
    sol[nPts] = ṽ_0
    # save initial condition
    if nStepsSave !== nothing
        ũ = sol[umap[1, :]]
        ṽ = sol[umap[2, :]]
        b̃ = sol[umap[3, :]]
        P̃x̃ = sol[nPts]
        iSave = 0
        saveCheckpoint1DTCNondim(ũ, ṽ, b̃, P̃x̃, t̃, iSave)
        iSave += 1
    end

    # main loop
    for i=1:nSteps
        t̃ += Δt̃

        # right-hand-side as a vector
        RHSVec = RHS*sol + diffVec

        # boundary conditions
        RHSVec[umap[1, 1]]  = 0  # u = 0 bot
        RHSVec[umap[1, nz̃]] = 0  # u decay top
        RHSVec[umap[2, 1]]  = 0  # v = 0 bot
        RHSVec[umap[2, nz̃]] = 0  # v decay top
        RHSVec[umap[3, 1]]  = -N^2 # b flux bot
        RHSVec[umap[3, nz̃]] = 0  # b flux top
        if canonical
            RHSVec[nPts] = ṽ_0 # ṽ_0 = P_x
        else
            RHSVec[nPts] = 0  # integral constraint
        end

        # solve
        sol = LHS\RHSVec

        # log
        if nStepsSave !== nothing && i % nStepsSave == 0
            #= println(@sprintf("t = %.1e (i = %d)", t, i)) =#

            # gather solution
            ũ = sol[umap[1, :]]
            ṽ = sol[umap[2, :]]
            b̃ = sol[umap[3, :]]
            P̃x̃ = sol[nPts]

            # save data
            saveCheckpoint1DTCNondim(ũ, ṽ, b̃, P̃x̃, t̃, iSave)
            iSave += 1
        end
    end

    ũ = sol[umap[1, :]]
    ṽ = sol[umap[2, :]]
    b̃ = sol[umap[3, :]]
    P̃x̃ = sol[nPts]
    return ũ, ṽ, b̃, P̃x̃
end
