################################################################################
# PG inversion functions
################################################################################

"""
    inversionLHS = getInversionLHS(ν, f, H, σ)

Setup left hand side of linear system for problem.
"""
function getInversionLHS(ν::Array{Float64,1}, f::Float64, H::Float64, σ::Array{Float64,1})
    nσ = size(σ, 1)
    A = Tuple{Int64,Int64,Float64}[]  

    # for finite difference on the top and bottom boundary
    fd_bot = mkfdstencil(σ[1:3], σ[1], 1)
    # fd_top = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
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
    # b.c. 2: χ = U
    push!(A, (nσ-1, nσ,  1.0))

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
        
        # eqtn: dσσ(nu*dσσ(χ))/H^4 + f^2*χ/nu = rhs
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
    end

    # Create CSC sparse matrix from matrix elements
    inversionLHS = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nσ, nσ)

    return lu(inversionLHS)
end

"""
    inversionRHS = getInversionRHS(rhs, U)

Setup right hand side of linear system for problem.
"""
function getInversionRHS(rhs::Array{Float64,2}, U::Real)
    # boundary conditions
    rhs[:, [1, 2, end]] .= 0 # χ = 0, dσ(χ) = 0 at σ = -1, dσσ(χ) = 0 at σ = 0
    rhs[:, end-1] .= U       # χ = U at σ = 0
    return rhs
end

"""
    χ = computeχ(m, inversionRHS)

Compute inversion solution given right hand side `inversionRHS`.
"""
function computeχ(m::ModelSetup2DPG, inversionRHS::Array{Float64,2})
    # solve
    χ = zeros(m.nξ, m.nσ)
    for i=1:m.nξ
        χ[i, :] = m.inversionLHSs[i]\inversionRHS[i, :]
    end
    return χ
end
function computeχ(inversionLHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}, inversionRHS::Array{Float64,2})
    # solve
    χ = zeros(size(inversionRHS))
    for i=1:size(χ, 1)
        χ[i, :] = inversionLHSs[i]\inversionRHS[i, :]
    end
    return χ
end

"""
    U = computeU(m, χ_b)

Compute U such that it satisfies constraint equation derived from
island rule.
"""
function computeU(m::ModelSetup2DPG, χ_b::Array{Float64,2})
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
        term3[i] = trapz(m.f^2 ./(m.ν[i, :]).*(m.χ_U[i, :] .- 1), m.σ)*m.H[i]
    end
    term3 = sum(term3)/m.nξ
    
    # fourth term: ⟨(ν*χ_U_zz)_z⟩ at z = 0
    term4 = zeros(m.nξ)
    for i=1:m.nξ
        # ν*χ_zzz on the boundary
        term4[i] = m.ν[i, m.nσ]*differentiate_pointwise(m.χ_U[i, m.nσ-4:m.nσ], m.σ[m.nσ-4:m.nσ], m.σ[m.nσ], 3)/m.H[i]^3
        # ν_z*χ_zz on the boundary
        term4[i] += differentiate_pointwise(m.ν[i, m.nσ-2:m.nσ], m.σ[m.nσ-2:m.nσ], m.σ[m.nσ], 1)*differentiate_pointwise(m.χ_U[i, m.nσ-3:m.nσ], m.σ[m.nσ-3:m.nσ], m.σ[m.nσ], 2)/m.H[i]^3
    end
    term4 = sum(term4)/m.nξ

    return -(term1 + term2)/(term3 + term4)
end

"""
    uξ, uη, uσ, U = postProcess(m, χ)

Take streamfunction `χ` and compute `uξ`, `uη`, `uσ`, and `U`
from its definition. Computation is different depending on choice of coordinates.
"""
function postProcess(m::ModelSetup2DPG, χ::Array{Float64,2})
    # χ at σ = 0 is vertical integral of uξ
    U = χ[1, end] # just take first one since they all must be the same

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

    return uξ, uη, uσ, U
end

"""
    χ, uξ, uη, uσ, U = invert(m, b; bl=false)

Invert for flow given current model state buoyancy perturbation.
"""
function invert(m::ModelSetup2DPG, b::Array{Float64,2}; bl=false)
    # buoyancy solution: rhs = dx(b), U = 0;
    # (U = 1 solution `sol_U` is stored in ModelSetup2DPG struct)
    if m.ξVariation
        rhs = xDerivative(m, b)
    else
        rhs = -repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*σDerivative(m, b)
    end

    if bl # BL Solution
        # no need for dzzzz anymore!
        χ = @. m.ν/m.f^2*rhs
    else # Full Inversion
        # buoyancy solution
        inversionRHS = getInversionRHS(rhs, 0)
        χ_b = computeχ(m, inversionRHS)

        # compute U such that "island rule" is satisfied
        if symmetry
            U = 0
        else
            U = computeU(m, χ_b)
        end

        # linearity: solution = χ_b + U*χ_U
        χ = χ_b + U*m.χ_U
    end

    if m.coords == "cylindrical"
        # b.c.: no flow at ρ = 0
        χ[1, :] .= 0
    end
    uξ, uη, uσ, U = postProcess(m, χ)

    return χ, uξ, uη, uσ, U
end
function invert!(m::ModelSetup2DPG, s::ModelState2DPG; bl=false)
    χ, uξ, uη, uσ, U = invert(m, s.b; bl=bl)
    s.χ[:, :] = χ
    s.uξ[:, :] = uξ
    s.uη[:, :] = uη
    s.uσ[:, :] = uσ
end