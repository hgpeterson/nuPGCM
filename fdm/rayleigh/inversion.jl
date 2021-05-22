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
    # Lower boundary condition: χ = 0 
    push!(A, (1, 1, 1.0))

    # Upper boundary condition: χ - U = 0
    push!(A, (nσ, nσ,  1.0))
    push!(A, (nσ, iU, -1.0))

    # Interior nodes
    for j=2:nσ-1
        row = j

        # dσσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        
        # eqtn: (f^2 + r^2)/r * dσσ(χ))/H^2 = -dξ(b) + dx(H)*σ*dσ(b)/H
        push!(A, (row, j-1, (f^2 + r^2)/r * fd_σσ[1]/H^2))
        push!(A, (row, j,   (f^2 + r^2)/r * fd_σσ[2]/H^2))
        push!(A, (row, j+1, (f^2 + r^2)/r * fd_σσ[3]/H^2))
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
    inversionRHS[:, [1, nσ]] .= 0 

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

    # compute uη = -f*uξ/r
    uη = -f*uξ/r

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
    U = computeU(sol_b, sol_U)

Compute U such that it satisfies constraint equation derived from
island rule.
"""
function computeU(sol_b, sol_U)
    # unpack
    χ_b = sol_b[:, 1:nσ]
    χ_U = sol_U[:, 1:nσ]

    # first term: ⟨χ_b_z⟩ at z = 0
    term1 = zeros(nξ)
    for i=1:nξ
        term1[i] = differentiate_pointwise(χ_b[i, nσ-3:nσ], σ[nσ-3:nσ], σ[nσ], 1)/H(ξ[i])
    end
    term1 = sum(term1)/nξ

    # second term: ⟨χ_U_z⟩ at z = 0
    term2 = zeros(nξ)
    for i=1:nξ
        term2[i] = differentiate_pointwise(χ_U[i, nσ-3:nσ], σ[nσ-3:nσ], σ[nσ], 1)/H(ξ[i])
    end
    term2 = sum(term2)/nξ

    # U = -term1/term2
    return -term1/term2
end

"""
    χ, uξ, uη, uσ, U = invert(b)

Wrapper function that inverts for flow given buoyancy perturbation `b`.
"""
function invert(b)
    # buoyancy solution: rhs = -dx(b), U = 0
    # dx(b) = -dξ(b) + dx(H)*σ*dσ(b)/H
    if ξVariation
        rhs = -xDerivativeTF(b)
    else
        rhs = Hx.(ξξ).*σσ.*σDerivativeTF(b)./H.(ξξ)
    end
    inversionRHS = getInversionRHS(rhs, 0)
    sol_b = computeSol(inversionRHS)

    # particular solution (sol_U) is global variable computed in run.jl

    # compute U such that "island rule" is satisfied
    U = computeU(sol_b, sol_U)

    # linearity: solution = sol_b + U*sol_U
    χ, uξ, uη, uσ, U = postProcess(sol_b + U*sol_U)

    return χ, uξ, uη, uσ, U
end

"""
    b, u, v, w = pointwise1DConstantκ(t, inversionLHS)

Apply the 1D solution to the Rayleigh drag problem pointwise over the domain.
See CF18 for details.
"""
function pointwise1DConstantκ(t)
    # inverse boundary layer thickness
    q = @. sqrt(r*N^2*Hx(x)^2/(κ*(f^2 + r^2)))

    # time dependent analytical buoyancy solution (only works for constant κ)
    ẑ = @. (z + H(x))/cosθ # NOTE THE COSINE HERE TO FIX b AND v (see notes)
    b = @. N^2*cosθ/q*(exp(-q*ẑ) - 0.5*(exp(-q*ẑ)*erfc(q*sqrt(κ*t) - ẑ/2/sqrt(κ*t)) + exp(q*ẑ)*erfc(q*sqrt(κ*t) + ẑ/2/sqrt(κ*t))))

    # invert for flow using rotated 1D equations
    û = @. b*sinθ/((f^2 + r^2)*cosθ^2/r)
    v = @. -f*û*cosθ/r

    # rotate
    u = @. û*cosθ
    w = @. û*sinθ

    return b, u, v, w
end
