################################################################################
# PG inversion functions
################################################################################

"""
    inversion_LHS = get_inversion_LHS(ν, f, H, σ)

Setup left hand side of linear system for 2D inversion problem.
"""
function get_inversion_LHS(ν, f, H, σ)
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
    inversion_LHS = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nσ, nσ)

    return lu(inversion_LHS)
end

"""
    inversionRHS = getInversionRHS(rhs, U)

Setup right hand side of linear system for 2D inversion problem.
"""
function get_inversion_RHS(rhs, U)
    # boundary conditions
    rhs[:, [1, 2, end]] .= 0 # χ = 0, dσ(χ) = 0 at σ = -1, dσσ(χ) = 0 at σ = 0
    rhs[:, end-1] .= U       # χ = U at σ = 0
    return rhs
end

"""
    χ = get_χ(m, inversionRHS)

Compute inversion solution given right hand side `inversion_RHS`.
"""
function get_χ(m::ModelSetup2D, inversion_RHS)
    return get_χ(m.inversion_LHSs, inversion_RHS)
end
function get_χ(inversion_LHSs, inversion_RHS)
    # solve
    χ = zeros(size(inversion_RHS))
    for i ∈ axes(χ, 1)
        χ[i, :] = inversion_LHSs[i]\inversion_RHS[i, :]
    end
    return χ
end

"""
    U = get_U(m, χ_b)

Compute U such that it satisfies constraint equation derived from
island rule.
"""
function get_U(m::ModelSetup2D, χ_b)
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
function get_U_BL(m::ModelSetup2D, b)
    # bx
    bx = ∂x(m, b)

    # first term: ⟨ ∫ ∂ₓb dσ ⟩ 
    term1 = zeros(m.nξ)
    for i=1:m.nξ
        term1[i] = m.H[i]*trapz(bx[i, :], m.σ)
    end
    term1 = sum(term1)/m.nξ

    # second term: ⟨ 1/q * 1/(1 + μS) * ∂ξ(b)(-1) ⟩
    dbdξ = ∂ξ(m, b)[:, 1]
    δ = @. sqrt(2*m.ν[:, 1]/abs(m.f))
    μ = @. m.ν[:, 1]/m.κ[:, 1]
    S = @. -1/m.f^2 * m.Hx*dbdξ
    q = @. 1/δ * (1 + μ*S)^(1/4)
    term2 = @. 1/q * 1/(1 + μ*S) * dbdξ
    term2 = sum(term2)/m.nξ

    # third term: ⟨ f^2/q/ν(-1) * 1/(1 + μS) ⟩
    term3 = @. m.f^2/q/m.ν[:, 1] * 1/(1 + μ*S)
    term3 = sum(term3)/m.nξ

    return (term1 + term2)/term3
end

"""
    uξ, uη, uσ, U = post_process(m, χ)

Take streamfunction `χ` and compute `uξ`, `uη`, `uσ`, and `U`
from its definition. Computation is different depending on choice of coordinates.
"""
function post_process(m::ModelSetup2D, χ)
    # χ at σ = 0 is vertical integral of uξ
    U = χ[1, end] # just take first one since they all must be the same

    # uξ = dσ(χ)/H
    uξ = ∂σ(m, χ)./repeat(m.H, 1, m.nσ)

    # uη = int_-1^0 f*χ/nu dσ*H
    uη = zeros(m.nξ, m.nσ)
    for i=1:m.nξ
        uη[i, :] = cumtrapz(m.f*(χ[i, :] .- U)./(m.ν[i, :]), m.σ)*m.H[i]
    end

    if m.coords == "cartesian"
        # uσ = -dξ(χ)/H
        uσ = -∂ξ(m, χ)./repeat(m.H, 1, m.nσ)
    elseif m.coords == "axisymmetric"
        # uσ = -dρ(ρ*χ)/(H*ρ)
        uσ = -∂ξ(m, repeat(m.ξ, 1, m.nσ).*χ)./repeat(m.H.*m.ξ, 1, m.nσ)
        # assume χ = 0 at ρ = 0
        fd_ξ = mkfdstencil([0, m.ξ[1], m.ξ[2]], m.ξ[1], 1)
        uσ[1, :] = @. -(fd_ξ[2]*m.ξ[1]*χ[1, :] + fd_ξ[3]*m.ξ[2]*χ[2, :])/(m.H[1]*m.ξ[1])
    end

    return uξ, uη, uσ, U
end

"""
    χ, uξ, uη, uσ, U = invert(m, b)

Invert for flow given current model state buoyancy perturbation.
"""
function invert(m::ModelSetup2D, b)
    # buoyancy solution: rhs = dx(b), U = 0;
    # (U = 1 solution `sol_U` is stored in ModelSetup2DPG struct)
    rhs = ∂x(m, b)

    if m.bl # BL Solution
        # bl inversion
        χ_b = @. m.ν/m.f^2*rhs

        # get U
        if m.no_net_transport
            U = 0
        else
            U = get_U_BL(m, b)
        end

        # χ_U = 1 in interior
        χ = χ_b .+ U
    else # Full Inversion
        # buoyancy solution
        inversionRHS = get_inversion_RHS(rhs, 0)
        χ_b = get_χ(m, inversionRHS)

        # compute U such that "island rule" is satisfied
        if m.no_net_transport
            U = 0
        else
            U = get_U(m, χ_b)
        end

        # linearity: solution = χ_b + U*χ_U
        χ = χ_b + U*m.χ_U
    end

    uξ, uη, uσ, U = post_process(m, χ)

    return χ, uξ, uη, uσ, U
end
function invert!(m::ModelSetup2D, s::ModelState2D)
    χ, uξ, uη, uσ, U = invert(m, s.b)
    s.χ[:, :] = χ
    s.uξ[:, :] = uξ
    s.uη[:, :] = uη
    s.uσ[:, :] = uσ
end
