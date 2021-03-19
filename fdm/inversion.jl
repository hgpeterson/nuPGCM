################################################################################
# Functions used to compute the flow field given a buoyancy perturbation using
# finite differences, terrain-following coordinates, and taking advantage of 
# a 2D geometry.
################################################################################

"""
    LHS = getInversionLHS()

Setup left hand side of linear system for problem.
"""
function getInversionLHS(κ, H)
    iU = nσ + 1
    A = Tuple{Int64,Int64,Float64}[]  

    # for finite difference on the top and bottom boundary
    fd_bot = mkfdstencil(σ[1:3], σ[1], 1)
    fd_top = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    fd_top_σσ = mkfdstencil(σ[nσ-3:nσ], σ[nσ], 2)

    # Main loop, insert stencil in matrix for each node point
    # Lower boundary conditions 
    # b.c. 1: dσ(χ) = 0
    push!(A, (1, 1, fd_bot[1]))
    push!(A, (1, 2, fd_bot[2]))
    push!(A, (1, 3, fd_bot[3]))
    # b.c. 2: χ = 0 
    push!(A, (2, 1, 1.0))

    # Upper boundary conditions
    # b.c. 1: dσσ(χ) = 0 
    push!(A, (nσ, nσ-3, fd_top_σσ[1]))
    push!(A, (nσ, nσ-2, fd_top_σσ[2]))
    push!(A, (nσ, nσ-1, fd_top_σσ[3]))
    push!(A, (nσ, nσ,   fd_top_σσ[4]))
    # b.c. 2: χ - U = 0
    push!(A, (nσ-1, nσ,  1.0))
    push!(A, (nσ-1, iU, -1.0))

    # Interior nodes
    for j=3:nσ-2
        row = j

        # dσ stencil
        fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
        κ_σ = sum(fd_σ.*κ[j-1:j+1])

        # dσσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        κ_σσ = sum(fd_σσ.*κ[j-1:j+1])

        # dσσσ stencil
        fd_σσσ = mkfdstencil(σ[j-2:j+2], σ[j], 3)

        # dσσσσ stencil
        fd_σσσσ = mkfdstencil(σ[j-2:j+2], σ[j], 4)
        
        # eqtn: dσσ(nu*dσσ(χ))/H^4 + f^2*(χ - U)/nu = dξ(b) - dx(H)*σ*dσ(b)/H
        # term 1 (product rule)
        push!(A, (row, j-1, Pr*κ_σσ*fd_σσ[1]/H^4))
        push!(A, (row, j,   Pr*κ_σσ*fd_σσ[2]/H^4))
        push!(A, (row, j+1, Pr*κ_σσ*fd_σσ[3]/H^4))

        push!(A, (row, j-2, 2*Pr*κ_σ*fd_σσσ[1]/H^4))
        push!(A, (row, j-1, 2*Pr*κ_σ*fd_σσσ[2]/H^4))
        push!(A, (row, j,   2*Pr*κ_σ*fd_σσσ[3]/H^4))
        push!(A, (row, j+1, 2*Pr*κ_σ*fd_σσσ[4]/H^4))
        push!(A, (row, j+2, 2*Pr*κ_σ*fd_σσσ[5]/H^4))

        push!(A, (row, j-2, Pr*κ[j]*fd_σσσσ[1]/H^4))
        push!(A, (row, j-1, Pr*κ[j]*fd_σσσσ[2]/H^4))
        push!(A, (row, j,   Pr*κ[j]*fd_σσσσ[3]/H^4))
        push!(A, (row, j+1, Pr*κ[j]*fd_σσσσ[4]/H^4))
        push!(A, (row, j+2, Pr*κ[j]*fd_σσσσ[5]/H^4))
        # term 2
        push!(A, (row, j,   f^2/(Pr*κ[j])))
        push!(A, (row, iU, -f^2/(Pr*κ[j])))
    end

    # U equation: U = ?
    row = iU
    push!(A, (row, row, 1.0))

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nσ+1, nσ+1)

    return A
end

"""
    RHS = getInversionRHS(rhs, U)

Setup right hand side of linear system for problem.
"""
function getInversionRHS(rhs, U)
    # last row is for U
    inversionRHS = zeros(nξ, nσ+1)
    iU = nσ + 1

    # fill rhs for interior nodes
    inversionRHS[:, 1:nσ] = rhs

    # zeros for boundary conditions
    inversionRHS[:, [1, 2, nσ-1, nσ]] .= 0 

    # U = ?
    inversionRHS[:, iU] .= U

    return inversionRHS
end

"""
    χ, uξ, uη, uσ, U = postProcess(sol)

Take solution `sol` and extract reshaped `χ` and `U`. Compute `uξ`, `uη`, `uσ` 
from definition of χ.
"""
function postProcess(sol)
    iU = nσ + 1

    # χ at σ = 0 is vertical integral of uξ
    U = sol[1, iU] # just take first one since they all must be the same

    # rest of solution is χ
    χ = sol[:, 1:nσ]

    # compute uξ = dσ(χ)/H
    uξ = σDerivativeTF(χ)./H.(x)

    # compute uη = int_-1^0 f*χ/nu dσ*H
    uη = zeros(nξ, nσ)
    for i=1:nξ
        uη[i, :] = cumtrapz(f*(χ[i, :] .- U)./(Pr*κ[i, :]), σ)*H(ξ[i])
    end

    # compute uσ = -dξ(χ)/H
    if ξVariation
        uσ = -ξDerivativeTF(χ)./H.(x)
    else
        uσ = zeros(nξ, nσ)
    end

    return χ, uξ, uη, uσ, U
end

"""
    sol = computeSol(inversionRHS)

Compute inversion solution given right hand side `inversionRHS`.
"""
function computeSol(inversionRHS)
    # solve
    sol = zeros(nξ, nσ+1)
    for i=1:nξ
        sol[i, :] = inversionLHSs[i]\inversionRHS[i, :]
    end
    return sol
end

"""
    U = computeU(solʰ, solᵖ)

Compute U such that it satisfies constraint equation derived from
island rule.
"""
function computeU(solʰ, solᵖ)
    # unpack
    χʰ = solʰ[:, 1:nσ]
    χᵖ = solᵖ[:, 1:nσ]

    # first term: ⟨(ν*χʰ_zz)_z⟩ at z = 0
    #= term1 = zDerivativeTF(Pr*κ .*zDerivativeTF(zDerivativeTF(χʰ))) =#
    #= term1 = term1[:, nσ] =#
    term1 = zeros(nξ)
    for i=1:nξ
        # χ_zzz on the boundary
        term1[i] = Pr*κ[i, nσ]*differentiate_pointwise(χʰ[i, nσ-4:nσ], σ[nσ-4:nσ], σ[nσ], 3)/H(ξ[i])^3
        # κ_z*χ_zz on the boundary
        term1[i] += Pr*differentiate_pointwise(κ[i, nσ-2:nσ], σ[nσ-2:nσ], σ[nσ], 1)*differentiate_pointwise(χʰ[i, nσ-3:nσ], σ[nσ-3:nσ], σ[nσ], 2)/H(ξ[i])^3
    end
    term1 = sum(term1)/nξ

    # second term: ⟨∫f^2/ν*χʰ⟩    
    term2 = zeros(nξ)
    for i=1:nξ
        term2[i] = trapz(f^2 ./(Pr*κ[i, :]).*χʰ[i, :], σ)*H(ξ[i])
    end
    term2 = sum(term2)/nξ

    # third term: ⟨∫f^2/ν*(χᵖ-1)⟩    
    term3 = zeros(nξ)
    for i=1:nξ
        term3[i] = trapz(f^2 ./(Pr*κ[i, :]).*(χᵖ[i, :] .- 1), σ)*H(ξ[i])
    end
    term3 = sum(term3)/nξ
    
    # fourth term: ⟨(ν*χᵖ_zz)_z⟩ at z = 0
    #= term4 = zDerivativeTF(Pr*κ .*zDerivativeTF(zDerivativeTF(χᵖ))) =#
    #= term4 = term4[:, nσ] =#
    term4 = zeros(nξ)
    for i=1:nξ
        # χ_zzz on the boundary
        term4[i] = Pr*κ[i, nσ]*differentiate_pointwise(χᵖ[i, nσ-4:nσ], σ[nσ-4:nσ], σ[nσ], 3)/H(ξ[i])^3
        # κ_z*χ_zz on the boundary
        term4[i] += Pr*differentiate_pointwise(κ[i, nσ-2:nσ], σ[nσ-2:nσ], σ[nσ], 1)*differentiate_pointwise(χᵖ[i, nσ-3:nσ], σ[nσ-3:nσ], σ[nσ], 2)/H(ξ[i])^3
    end
    term4 = sum(term4)/nξ

    return -(term1 + term2)/(term3 + term4)
end

"""
    χ, uξ, uη, uσ, U = invert(b)

Wrapper function that inverts for flow given buoyancy perturbation `b`.
"""
function invert(b)
    # homogeneous solution: rhs = dx(b), U = 0
    # dx(b) = dξ(b) - dx(H)*σ*dσ(b)/H
    if ξVariation
        rhs = xDerivativeTF(b)
    else
        rhs = -Hx.(ξξ).*σσ.*σDerivativeTF(b)./H.(ξξ)
    end
    inversionRHS = getInversionRHS(rhs, 0)
    solʰ = computeSol(inversionRHS)

    # particular solution is global variable computed in runPGSolver.jl

    # compute U such that "island rule" is satisfied
    U = computeU(solʰ, solᵖ)
    println(@sprintf("U = %1.1e m2 s-1", U))

    # linearity: solution = solʰ + U*solᵖ
    χ, uξ, uη, uσ, U = postProcess(solʰ + U*solᵖ)

    return χ, uξ, uη, uσ, U
end

"""
    χEkman = getChiEkman(b)

Compute Ekman layer solution to problem given buoyancy perturbation b.
"""
function getChiEkman(b)
    # compute x derivative of b
    bx = xDerivativeTF(b)

    # Ekman layer thickness
    δ = sqrt(2*Pr*κ1/abs(f)) # using κ at the bottom

    # interior solution: thermal wind balance
    χ_I = bx
    χ_I_bot = repeat(χ_I[:, 1], 1, nσ)
    χ_I_top = repeat(χ_I[:, nσ], 1, nσ)

    # bottom Ekman layer correction
    χ_B_bot = @. -exp(-(z + H(x))/δ)*χ_I_bot*(cos((z + H(x))/δ) + sin((z + H(x))/δ))

    # top Ekman layer correction
    χ_B_top = @. -exp(z/δ)*χ_I_top*cos(z/δ)

    # full solution (use full κ with assumption that its variation is larger than δ)
    χEkman = @. Pr*κ/f^2*(χ_I + χ_B_bot + χ_B_top)

    return χEkman
end
