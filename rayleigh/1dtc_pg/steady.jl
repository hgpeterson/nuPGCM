"""
    A, rhs = getMatrices()   

Compute matrices for 1D equations.
"""
function getMatrices()
    nVars = 3
    nPts = nVars*nẑ

    umap = reshape(1:nPts, nVars, nẑ)    
    A = Tuple{Int64,Int64,Float64}[] # LHS matrix 
    rhs = zeros(nPts)                # RHS vector

    # Main loop, insert stencil in matrices for each node point
    for j=1:nẑ
        # eqtn 1: -f*v̂*cos(θ) - b*sin(θ) + r*û*cos(θ)^2
        row = umap[1, j]
        push!(A, (row, umap[2, j],   -f*cos(θ)))
        push!(A, (row, umap[3, j],   -sin(θ)))
        push!(A, (row, umap[1, j],   r*cos(θ)^2))

        # eqtn 2: f*û*cos(θ) + rv̂ = 0 
        row = umap[2, j]
        push!(A, (row, umap[1, j],   f*cos(θ)))
        push!(A, (row, umap[2, j],   r))
    end

    for j=2:nẑ-1
        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 1)
        κ_ẑ = sum(fd_ẑ.*κ[j-1:j+1])

        # dẑẑ stencil
        fd_ẑẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 2)


        # eqtn 3: N^2*û*sin(θ) - dẑ(κ*dẑ(b)) = dẑ(κ)*N^2*cos(θ) 
        row = umap[3, j]
        push!(A, (row, umap[1, j],   N^2*sin(θ)))
        push!(A, (row, umap[3, j-1], -(κ_ẑ*fd_ẑ[1] + κ[j]*fd_ẑẑ[1])))
        push!(A, (row, umap[3, j],   -(κ_ẑ*fd_ẑ[2] + κ[j]*fd_ẑẑ[2])))
        push!(A, (row, umap[3, j+1], -(κ_ẑ*fd_ẑ[3] + κ[j]*fd_ẑẑ[3])))
        rhs[row] = κ_ẑ*N^2*cos(θ)
    end

    # Boundary Conditions: Bottom
    # dẑ(b) = -N^2*cos(θ)
    row = umap[3, 1] 
    fd_ẑ = mkfdstencil(ẑ[1:3], ẑ[1], 1)
    push!(A, (row, umap[3, 1], fd_ẑ[1]))
    push!(A, (row, umap[3, 2], fd_ẑ[2]))
    push!(A, (row, umap[3, 3], fd_ẑ[3]))

    # Boundary Conditions: Top
    fd_ẑ = mkfdstencil(ẑ[nẑ-2:nẑ], ẑ[nẑ], 1)
    # dẑ(b) = 0
    row = umap[3, nẑ]
    push!(A, (row, umap[3, nẑ-2], fd_ẑ[1]))
    push!(A, (row, umap[3, nẑ-1], fd_ẑ[2]))
    push!(A, (row, umap[3, nẑ],   fd_ẑ[3]))

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
    nPts = nVars*nẑ

    # for flattening for matrix mult
    umap = reshape(1:nPts, nVars, nẑ)

    # get matrices and vectors
    A, rhs = getMatrices()

    # boundaries
    rhs[umap[3, 1]]  = -N^2*cos(θ) # b flux bot
    rhs[umap[3, nẑ]] = 0 # b flux top

    # solve
    solVec = A\rhs

    # gather solution and rotate
    sol = reshape(solVec, 3, nẑ)
    û = sol[1, :]
    v̂ = sol[2, :]
    b = sol[3, :]

    # compute χ and U
    χ = cumtrapz(û, ẑ)
    U = trapz(û, ẑ)

    # save data
    saveCheckpoint1DTCPGRayleigh(b, χ, û, v̂, U, -42, 999)

    return b
end