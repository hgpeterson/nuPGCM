################################################################################
# PG inversion functions
################################################################################

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
function computeSol(inversionLHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}, inversionRHS::Array{Float64,2})
    # solve
    sol = zeros(size(inversionRHS))
    for i=1:size(sol, 1)
        sol[i, :] = inversionLHSs[i]\inversionRHS[i, :]
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
from definition of χ. Computation is different depending on choice of coordinates.
"""
function postProcess(m::ModelSetup, sol::Array{Float64,2})
    iU = m.nσ + 1

    # χ at σ = 0 is vertical integral of uξ
    U = sol[1, iU] # just take first one since they all must be the same

    # rest of solution is χ
    χ = sol[:, 1:m.nσ]

    # uξ = dσ(χ)/H
    uξ = σDerivative(m, χ)./repeat(m.H, 1, m.nσ)

    # uη = int_-1^0 f*χ/nu dσ*H
    uη = zeros(m.nξ, m.nσ)
    for i=1:m.nξ
        uη[i, :] = cumtrapz(m.f*(χ[i, :] .- U)./(m.ν[i, :]), m.σ)*m.H[i]
    end

    if m.ξVariation
        if m.coords == "cartesian"
            # uσ = -dξ(χ)/H
            uσ = -ξDerivative(m, χ)./repeat(m.H, 1, m.nσ)
        elseif m.coords == "cylindrical"
            # uσ = -dρ(ρ*χ)/(H*ρ)
            uσ = -ξDerivative(m, repeat(m.ξ, 1, m.nσ).*χ)./repeat(m.H.*m.ξ, 1, m.nσ)
        end
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
        rhs = xDerivative(m, b)
    else
        rhs = -repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*σDerivative(m, b)
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
    χ, uξ, uη, uσ, U = invert(m, s.b; bl=bl)
    s.χ[:, :] = χ
    s.uξ[:, :] = uξ
    s.uη[:, :] = uη
    s.uσ[:, :] = uσ
end