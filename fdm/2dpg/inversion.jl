################################################################################
# Functions used to compute the flow field given a buoyancy perturbation using
# finite differences, terrain-following coordinates, and taking advantage of 
# a 2D geometry.
################################################################################

"""
    LHS = getInversionLHS()

Setup left hand side of linear system for problem.
"""
function getInversionLHS(ν::Array{Float64,1}, f::Float64,H::Float64,σ::Array{Float64,1})
    nσ = size(σ, 1)
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
        ν_σ = sum(fd_σ.*ν[j-1:j+1])

        # dσσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        ν_σσ = sum(fd_σσ.*ν[j-1:j+1])

        # dσσσ stencil
        fd_σσσ = mkfdstencil(σ[j-2:j+2], σ[j], 3)

        # dσσσσ stencil
        fd_σσσσ = mkfdstencil(σ[j-2:j+2], σ[j], 4)
        
        # eqtn: dσσ(nu*dσσ(χ))/H^4 + f^2*(χ - U)/nu = dξ(b) - dx(H)*σ*dσ(b)/H
        # term 1 (product rule)
        push!(A, (row, j-1, ν_σσ*fd_σσ[1]/H^4))
        push!(A, (row, j,   ν_σσ*fd_σσ[2]/H^4))
        push!(A, (row, j+1, ν_σσ*fd_σσ[3]/H^4))

        push!(A, (row, j-2, 2*ν_σ*fd_σσσ[1]/H^4))
        push!(A, (row, j-1, 2*ν_σ*fd_σσσ[2]/H^4))
        push!(A, (row, j,   2*ν_σ*fd_σσσ[3]/H^4))
        push!(A, (row, j+1, 2*ν_σ*fd_σσσ[4]/H^4))
        push!(A, (row, j+2, 2*ν_σ*fd_σσσ[5]/H^4))

        push!(A, (row, j-2, ν[j]*fd_σσσσ[1]/H^4))
        push!(A, (row, j-1, ν[j]*fd_σσσσ[2]/H^4))
        push!(A, (row, j,   ν[j]*fd_σσσσ[3]/H^4))
        push!(A, (row, j+1, ν[j]*fd_σσσσ[4]/H^4))
        push!(A, (row, j+2, ν[j]*fd_σσσσ[5]/H^4))
        # term 2
        push!(A, (row, j,   f^2/(ν[j])))
        push!(A, (row, iU, -f^2/(ν[j])))
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
function getInversionRHS(rhs::Array{Float64,2}, U::Real)
    # get shape
    nξ = size(rhs, 1)
    nσ = size(rhs, 2)

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
    sol = computeSol(m, inversionRHS)

Compute inversion solution given right hand side `inversionRHS`.
"""
function computeSol(m::ModelSetup, inversionRHS::Array{Float64,2})
    # solve
    sol = zeros(m.nξ, m.nσ+1)
    for i=1:m.nξ
        sol[i, :] = m.inversionLHSs[i]\inversionRHS[i, :]
    end
    return sol
end

"""
    U = computeU(m, sol_b)

Compute U such that it satisfies constraint equation derived from
island rule.
"""
function computeU(m::ModelSetup, sol_b::Array{Float64,2})
    # unpack
    χ_b = sol_b[:, 1:m.nσ]
    χ_U = m.sol_U[:, 1:m.nσ]

    # first term: ⟨(ν*χ_b_zz)_z⟩ at z = 0
    term1 = zeros(m.nξ)
    for i=1:m.nξ
        # ν*χ_zzz on the boundary
        term1[i] = m.ν[i, m.nσ]*differentiate_pointwise(χ_b[i, m.nσ-4:m.nσ], m.σ[m.nσ-4:m.nσ], m.σ[m.nσ], 3)/m.H[i]^3
        # ν_z*χ_zz on the boundary
        term1[i] += differentiate_pointwise(m.ν[i, m.nσ-2:m.nσ], m.σ[m.nσ-2:m.nσ], m.σ[m.nσ], 1)*differentiate_pointwise(χ_b[i, m.nσ-3:m.nσ], m.σ[m.nσ-3:m.nσ], m.σ[m.nσ], 2)/m.H[i]^3
    end
    term1 = sum(term1)/m.nξ

    # second term: ⟨∫f^2/ν*χ_b⟩    
    term2 = zeros(m.nξ)
    for i=1:m.nξ
        term2[i] = trapz(m.f^2 ./(m.ν[i, :]).*χ_b[i, :], m.σ)*m.H[i]
    end
    term2 = sum(term2)/m.nξ

    # third term: ⟨∫f^2/ν*(χ_U-1)⟩    
    term3 = zeros(m.nξ)
    for i=1:m.nξ
        term3[i] = trapz(m.f^2 ./(m.ν[i, :]).*(χ_U[i, :] .- 1), m.σ)*m.H[i]
    end
    term3 = sum(term3)/m.nξ
    
    # fourth term: ⟨(ν*χ_U_zz)_z⟩ at z = 0
    term4 = zeros(m.nξ)
    for i=1:m.nξ
        # ν*χ_zzz on the boundary
        term4[i] = m.ν[i, m.nσ]*differentiate_pointwise(χ_U[i, m.nσ-4:m.nσ], m.σ[m.nσ-4:m.nσ], m.σ[m.nσ], 3)/m.H[i]^3
        # ν_z*χ_zz on the boundary
        term4[i] += differentiate_pointwise(m.ν[i, m.nσ-2:m.nσ], m.σ[m.nσ-2:m.nσ], m.σ[m.nσ], 1)*differentiate_pointwise(χ_U[i, m.nσ-3:m.nσ], m.σ[m.nσ-3:m.nσ], m.σ[m.nσ], 2)/m.H[i]^3
    end
    term4 = sum(term4)/m.nξ

    return -(term1 + term2)/(term3 + term4)
end

"""
    χ, uξ, uη, uσ, U = postProcess(m, sol)

Take solution `sol` and extract reshaped `χ` and `U`. Compute `uξ`, `uη`, `uσ` 
from definition of χ.
"""
function postProcess(m::ModelSetup, sol::Array{Float64,2})
    iU = m.nσ + 1

    # χ at σ = 0 is vertical integral of uξ
    U = sol[1, iU] # just take first one since they all must be the same

    # rest of solution is χ
    χ = sol[:, 1:m.nσ]

    # compute uξ = dσ(χ)/H
    uξ = σDerivativeTF(m, χ)./repeat(m.H, 1, m.nσ)

    # compute uη = int_-1^0 f*χ/nu dσ*H
    uη = zeros(m.nξ, m.nσ)
    for i=1:m.nξ
        uη[i, :] = cumtrapz(m.f*(χ[i, :] .- U)./(m.ν[i, :]), m.σ)*m.H[i]
    end

    # compute uσ = -dξ(χ)/H
    if m.ξVariation
        uσ = -ξDerivativeTF(m, χ)./repeat(m.H, 1, m.nσ)
    else
        uσ = zeros(m.nξ, m.nσ)
    end

    return χ, uξ, uη, uσ, U
end

"""
    χ, uξ, uη, uσ, U = invert(m, b; bl=false)

Invert for flow given current model state buoyancy perturbation.
"""
function invert(m::ModelSetup, b::Array{Float64,2}; bl=false)
    # buoyancy solution: rhs = dx(b), U = 0;
    # (U = 1 solution `sol_U` is stored in ModelSetup struct)
    if m.ξVariation
        rhs = xDerivativeTF(m, b)
    else
        rhs = -repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*σDerivativeTF(m, b)
    end

    if bl # BL Solution
        # no need for dzzzz anymore!
        χ = @. m.ν/m.f^2*rhs

        # pass sol array to postProcess
        sol = zeros(m.nξ, m.nσ + 1)
        sol[:, 1:m.nσ] = χ
    else # Full Inversion
        # buoyancy solution
        inversionRHS = getInversionRHS(rhs, 0)
        sol_b = computeSol(m, inversionRHS)

        # compute U such that "island rule" is satisfied
        if symmetry
            U = 0
        else
            U = computeU(m, sol_b)
        end

        # linearity: solution = sol_b + U*sol_U
        sol = sol_b + U*m.sol_U
    end

    χ, uξ, uη, uσ, U = postProcess(m, sol)

    return χ, uξ, uη, uσ, U
end
function invert!(m::ModelSetup, s::ModelState; bl=false)
    χ, uξ, uη, uσ, U = invert(m, s.b; bl)
    s.χ[:, :] = χ
    s.uξ[:, :] = uξ
    s.uη[:, :] = uη
    s.uσ[:, :] = uσ
end
