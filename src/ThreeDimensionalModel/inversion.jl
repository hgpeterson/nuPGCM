"""
    barotropic_LHS = get_barotropic_LHS(g, ρ₀, f₀, β, H, Hx, Hy, r_sym, r_asym)

Construct LHS matrix for barotropic vorticity equation with Dirichlet b.c. Ψ = 0. 
Returns LU-factored matrix.
"""
function get_barotropic_LHS(g1::G, g::G, ε², β, H::F, Hx::F, Hy::F, r_sym::F, r_asym::F) where {F <: FEField, G <: FEGrid}
    J = Jacobians(g1)
    w, ξ = quad_weights_points(4, 2)

    # create global linear system using stamping method
    A = Tuple{Int64,Int64,Float64}[]
    @showprogress "Building barotropic LHS..." for k=1:g.nt
        dξ = J.dets[k]
        ξx = J.Js[k, 1, 1]
        ξy = J.Js[k, 1, 2]
        ηx = J.Js[k, 2, 1]
        ηy = J.Js[k, 2, 2]

        # calculate contribution to K from element k
        K = zeros(g.nn, g.nn)
        for i=1:g.nn
            for j=1:g.nn
                func(ξ) = -evaluate(r_sym, transform_from_ref_el(ξ, g1.p[g1.t[k, :], :]), k)*
                            (∂φ(g.s, j, 1, ξ)*∂φ(g.s, i, 1, ξ)*(ξx^2 + ξy^2) + 
                             ∂φ(g.s, j, 1, ξ)*∂φ(g.s, i, 2, ξ)*(ξx*ηx + ξy*ηy) + 
                             ∂φ(g.s, j, 2, ξ)*∂φ(g.s, i, 1, ξ)*(ηx*ξx + ηy*ξy) +
                             ∂φ(g.s, j, 2, ξ)*∂φ(g.s, i, 2, ξ)*(ηx^2 + ηy^2))*dξ
                K[i, j] = ref_el_quad(func, w, ξ)
            end
        end

        # # calculate contribution to K′ from element k
        # K′ᵏ = zeros(FT, n, n)
        # for i=1:n
        #     for j=1:n
        #         func(ξ, η) = r_asym_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)^3*
        #                     (shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dξ=1) - 
        #                      shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η; dη=1))
        #         K′ᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
        #     end
        # end

        # # calculate contribution to C from element k
        # Cᵏ = zeros(FT, n, n)
        # for i=1:n
        #     for j=1:n
        #         func(ξ, η) = (β/H_func(ξ, η, k) - (f₀ + β*η)*Hy_func(ξ, η, k)/H_func(ξ, η, k)^2)*
        #                     shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η) -
        #                     -(f₀ + β*η)*Hx_func(ξ, η, k)/H_func(ξ, η, k)^2*
        #                     shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η)
        #         Cᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
        #     end
        # end

        # add to global system
        for i=1:g.nn
            for j=1:g.nn
                if g.t[k, i] ∈ g.e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
                # push!(A, (t[k, i], t[k, j], K′ᵏ[i, j]))
                # push!(A, (t[k, i], t[k, j], Cᵏ[i, j]))
            end
        end
    end

    # dirichlet Ψ = 0 along edges
    for i=1:g.ne
        push!(A, (g.e[i], g.e[i], 1))
    end

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), g.np, g.np)

    return lu(A)
end

"""
    barotropic_RHS = get_barotropic_RHS(m, τξ, τη)

Construct RHS vector for barotropic vorticity equation with b.c. Ψ = 0.
"""
function get_barotropic_RHS(g1::G, g::G, ρ₀, γ::F, τξ::F, τη::F, H::F, Hx::F, Hy::F) where {G <: FEGrid, F <: FEField}
    J = Jacobians(g1)
    w, ξ = quad_weights_points(4, 2)

    # functions
    H_func(ξ, η, k)   = fem_evaluate(m, m.H,  ξ, η, k)
    Hx_func(ξ, η, k)  = fem_evaluate(m, m.Hx, ξ, η, k)
    Hy_func(ξ, η, k)  = fem_evaluate(m, m.Hy, ξ, η, k)
    τξ_func(ξ, η, k)  = fem_evaluate(m, τξ,   ξ, η, k)
    τη_func(ξ, η, k)  = fem_evaluate(m, τη,   ξ, η, k)
    curl_τ(ξ, η, k)   = ∂ξ(m, τη, ξ, η, k)/H_func(ξ, η, k) - τη_func(ξ, η, k)/H_func(ξ, η, k)^2*Hx_func(ξ, η, k) -
                       (∂η(m, τξ, ξ, η, k)/H_func(ξ, η, k) - τξ_func(ξ, η, k)/H_func(ξ, η, k)^2*Hy_func(ξ, η, k))
    JEBAR(ξ, k) = 0

	# stamp
    RHS = zeros(g.np)
	@showprogress "Building barotropic RHS..." for k=1:g.nt
        dξ = J.dets[k]
        ξx = J.Js[k, 1, 1]
        ξy = J.Js[k, 1, 2]
        ηx = J.Js[k, 2, 1]
        ηy = J.Js[k, 2, 2]

        for i=1:g.nn
            if g.t[k, i] in g.e
                # edge node, leave as zero so that Ψ = 0
                continue
            end
            func(ξ) = (-JEBAR(ξ, k) + curl_τ(ξ, k)/ρ₀)*φ(g.s, i, ξ)*dξ
            RHS[g.t[k, i]] += ref_el_quad(func, w, ξ)
        end
	end

    return RHS
end

"""
    baroclinic_LHS = get_baroclinic_LHS(ρ₀, ν, f, H, σ)

Construct discrete LHS matrix for baroclinic problem
    ν/f/H² ∂σσ(τξ) + τη = rhs_x
    ν/f/H² ∂σσ(τη) - τξ = rhs_y
with boundary conditions τ(0) = τ₀ and -H² ∫ στ/ρ₀/ν dσ = U. Returns LU-factored matrix.
"""
function get_baroclinic_LHS(ρ₀::FT, ν::AbstractVector{FT}, f::FT, H::FT, σ::AbstractVector{FT}) where FT <: Real
    # convention: τξ is variable 1, τη is variable 2
    nσ = size(σ, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    A = Tuple{Int64,Int64,FT}[]  

    # Interior nodes
    for j=2:nσ-1 
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = rhs_x
        row = imap[1, j]
        # term 1
        push!(A, (row, imap[1, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(A, (row, imap[1, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(A, (row, imap[1, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(A, (row, imap[2, j], 1))

        # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = rhs_y
        row = imap[2, j]
        # term 1
        push!(A, (row, imap[2, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(A, (row, imap[2, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(A, (row, imap[2, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(A, (row, imap[1, j], -1))
    end

    # Upper boundary conditions: wind stress
    # b.c. 1: τξ = τξ₀ at σ = 0
    push!(A, (imap[1, nσ], imap[1, nσ], 1))
    # b.c. 2: τη = τη₀ at σ = 0
    push!(A, (imap[2, nσ], imap[2, nσ], 1))

    # Integral boundary conditions: transport
    # b.c. 1: -H² ∫ σ τξ/ρ₀/ν dσ = Uξ
    for j=1:nσ-1
        # trapezoidal rule
        push!(A, (imap[1, 1], imap[1, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(A, (imap[1, 1], imap[1, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end
    # b.c. 1: -H² ∫ σ τη/ρ₀/ν dσ = Uη
    for j=1:nσ-1
        # trapezoidal rule
        push!(A, (imap[2, 1], imap[2, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(A, (imap[2, 1], imap[2, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nvar*nσ, nvar*nσ)

    return lu(A)
end

"""
    baroclinic_RHS = get_baroclinic_RHS(rhs_x, rhs_y, τξ₀, τη₀, Uξ, Uη)

Construct discrete RHS vector for baroclinic problem
    ν/f/H² ∂σσ(τξ) + τη = rhs_x
    ν/f/H² ∂σσ(τη) - τξ = rhs_y
with boundary conditions τ(0) = τ₀ and -H² ∫ στ/ρ₀/ν dσ = U. 
"""
function get_baroclinic_RHS(rhs_x::AbstractVector{FT}, rhs_y::AbstractVector{FT}, 
                            τξ₀::Real, τη₀::Real, Uξ::Real, Uη::Real) where FT <: Real
    nσ = size(rhs_x, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    b = zeros(FT, nvar*nσ)

    # eqtns:
    # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = rhs_x
    b[imap[1, 2:nσ-1]] = rhs_x[2:nσ-1] 
    # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = rhs_y
    b[imap[2, 2:nσ-1]] = rhs_y[2:nσ-1] 

    # top b.c.:
    # b.c. 1: τξ = τξ₀ at σ = 0
    b[imap[1, nσ]] = τξ₀
    # b.c. 2: τη = τη₀ at σ = 0
    b[imap[2, nσ]] = τη₀
    
    # integral b.c.:
    # b.c. 1: -H² ∫ σ τξ/ρ₀/ν dσ = Uξ
    b[imap[1, 1]] = Uξ
    # b.c. 2: -H² ∫ σ τη/ρ₀/ν dσ = Uη
    b[imap[2, 1]] = Uη

    return b
end

"""
    τξ, τη = solve_baroclinic_systems(baroclinic_LHSs, baroclinic_RHSs)

Solve multiple linear baroclinic systems (in parallel someday?). 
"""
function solve_baroclinic_systems(baroclinic_LHSs::AbstractVector{SuiteSparse.UMFPACK.UmfpackLU}, 
                                  baroclinic_RHSs::AbstractMatrix{FT}) where FT <: Real
    np = size(baroclinic_RHSs, 1)
    nσ = Int64(size(baroclinic_RHSs, 2)/2)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    τξ = zeros(FT, np, nσ)
    τη = zeros(FT, np, nσ)
    for i=1:np
        sol = baroclinic_LHSs[i]\baroclinic_RHSs[i, :]
        τξ[i, :] = sol[imap[1, :]]
        τη[i, :] = sol[imap[2, :]]
    end
    return τξ, τη
end

"""
    H²τξ_b, H²τη_b = get_H²τ_b(m, b)

Compute the baroclinic stress due to buoyancy gradients.
"""
function get_H²τ_b(m::ModelSetup3DPG, b::AbstractMatrix{FT}) where FT <: Real
    # # analytical buoyancy gradients
    # rhs_x = zeros(FT, m.np, m.nσ)
    # rhs_y = zeros(FT, m.np, m.nσ)
    # N² = m.N²[1, 1] # constant
    # for j=1:m.nσ
    #     bξ = m.Hx./m.H.*b[:, j]
    #     bη = m.Hy./m.H.*b[:, j]
    #     bσ = N²*m.H*(1 - exp(-(m.σ[j] + 1)/0.1))
    #     bx = bξ - m.σ[j]*m.Hx./m.H.*bσ
    #     by = bη - m.σ[j]*m.Hy./m.H.*bσ
    #     rhs_x[:, j] = m.ρ₀*m.ν[:, j].*m.H.^2 ./(m.f₀ .+ m.β*m.p[:, 2]).*bx
    #     rhs_y[:, j] = m.ρ₀*m.ν[:, j].*m.H.^2 ./(m.f₀ .+ m.β*m.p[:, 2]).*by
    # end
    # baroclinic_RHSs_b = zeros(FT, m.np, 2*m.nσ)
    # for i=1:m.np
    #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
    # end
    # return solve_baroclinic_systems(m.baroclinic_LHSs, baroclinic_RHSs_b)

    # # integrals of buoyancy gradients on rhs
    # bσ_x = zeros(FT, m.np, m.nσ)
    # bσ_y = zeros(FT, m.np, m.nσ)
    # for i=1:m.np
    #     bσ_x[i, :] = -m.σ*m.Hx[i]/m.H[i].*differentiate(b[i, :], m.σ) 
    #     bσ_y[i, :] = -m.σ*m.Hy[i]/m.H[i].*differentiate(b[i, :], m.σ)
    # end
    # rhs_x = m.Cξ*b + m.M*bσ_x
    # rhs_y = m.Cη*b + m.M*bσ_y
    # println("b_x: ", maximum(abs.(rhs_x)))
    # println("M⁻¹b_x: ", maximum(abs.(m.M_LU\rhs_x)))
    # for i=1:m.np
    #     rhs_x[i, :] .*= m.ρ₀*m.ν[i, :]*m.H[i]^2/(m.f₀ + m.β*m.p[i, 2])
    #     rhs_y[i, :] .*= m.ρ₀*m.ν[i, :]*m.H[i]^2/(m.f₀ + m.β*m.p[i, 2])
    # end
    # # stress due to buoyancy gradients
    # baroclinic_RHSs_b = zeros(FT, m.np, 2*m.nσ)
    # for i=1:m.np
    #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
    # end
    # vξ_b, vη_b = solve_baroclinic_systems(m.baroclinic_LHSs, baroclinic_RHSs_b)
    # τξ_b = m.M_LU\vξ_b
    # τη_b = m.M_LU\vη_b
    # return τξ_b, τη_b

    # pointwise buoyancy gradients
    b_x = m.M_LU\(m.Cξ*b)
    b_y = m.M_LU\(m.Cη*b)
    for i=1:m.np
        b_x[i, :] += -m.σ*m.Hx[i].*differentiate(b[i, :], m.σ)/m.H[i] 
        b_y[i, :] += -m.σ*m.Hy[i].*differentiate(b[i, :], m.σ)/m.H[i]
    end
    println("b_x: ", maximum(abs.(b_x)))
    # stress due to buoyancy gradients
    baroclinic_RHSs_b = zeros(FT, m.np, 2*m.nσ)
    for i=1:m.np
        c = m.ρ₀*m.ν[i, :]*m.H[i]^2 ./ (m.f₀ .+ m.β*m.p[i, 2])
        baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(c.*b_x[i, :], c.*b_y[i, :], 0, 0, 0, 0)
    end
    return solve_baroclinic_systems(m.baroclinic_LHSs, baroclinic_RHSs_b)
end

"""
    H²τξ, H²τη = get_full_τ(m, H²τξ_b, H²τη_b, τξ₀, τη₀, Ψ)

Reconstruct full stress given the buoyancy, wind, and transport components.
"""
function get_full_τ(m::ModelSetup3DPG, H²τξ_b::AbstractMatrix{FT}, H²τη_b::AbstractMatrix{FT}, 
                    τξ₀::AbstractVector{FT}, τη₀::AbstractVector{FT}, Ψ::AbstractVector{FT}) where FT <: Real
    Uξ = -(m.M_LU\(m.Cη*Ψ))
    println("Uξ: ", maximum(abs.(Uξ)))
    Uη =  m.M_LU\(m.Cξ*Ψ)
    H²τξ = @. H²τξ_b + τξ₀*m.H²τξ_wξ - τη₀*m.H²τη_wξ + Uξ*m.H²τξ_tξ - Uη*m.H²τη_tξ
    H²τη = @. H²τη_b + τξ₀*m.H²τη_wξ + τη₀*m.H²τξ_wξ + Uξ*m.H²τη_tξ + Uη*m.H²τξ_tξ
    return H²τξ, H²τη
end

"""
    Huξ, Huη, Huσ = get_u(m, H²τξ, H²τη)

Compute velocity field given the baroclinic stresses.
"""
function get_u(m::ModelSetup3DPG, H²τξ::AbstractMatrix{FT}, H²τη::AbstractMatrix{FT}) where FT <: Real
    # integrate τξ and τη to get uξ and uη
    Huξ = zeros(FT, m.np, m.nσ)
    Huη = zeros(FT, m.np, m.nσ)
    for i=1:m.np
        Huξ[i, :] = 1/m.ρ₀*cumtrapz(H²τξ[i, :]./m.ν[i, :], m.σ)
        Huη[i, :] = 1/m.ρ₀*cumtrapz(H²τη[i, :]./m.ν[i, :], m.σ)
    end
    println("uξ: ", maximum(abs.(Huξ./m.H)))

    # take vertical derivative of continuity equation, then solve
    Dσσ = get_Dσσ(m.σ)
    rhs = -(m.M_LU\(m.Cξ*(H²τξ/m.ρ₀./m.ν) + m.Cη*(H²τη/m.ρ₀./m.ν)))
    rhs[:, 1] .= 0
    rhs[:, m.nσ] .= 0
    Huσ = zeros(FT, m.np, m.nσ)
    for i=1:m.np
        Huσ[i, :] = Dσσ\rhs[i, :]
    end

    return Huξ, Huη, Huσ
end

"""
    Ψ, uξ, uη, uσ = invert(m, τξ₀, τη₀, b)

Solve full 3D PG inversion equations given wind stress and buoyancy field.
"""
function invert(m::ModelSetup3DPG, τξ₀::AbstractVector{FT}, τη₀::AbstractVector{FT}, 
                b::AbstractMatrix{FT}; plots=false) where FT <: Real
    # solve for stress due to buoyancy gradients
    H²τξ_b, H²τη_b = get_H²τ_b(m, b)
    println("H²τξ_b: ", maximum(abs.(H²τξ_b)))

    # bottom stress 
    H²τξ_b_bot = H²τξ_b[:, 1]
    H²τη_b_bot = H²τη_b[:, 1]

    # buoyancy integral for JEBAR term
    γ = zeros(FT, m.np)
    for i=1:m.np
        γ[i] = -m.H[i]^2*trapz(m.σ.*b[i, :], m.σ)
    end

    # bottom stress due to wind stress
    H²τξ_wξ_bot = m.H²τξ_wξ[:, 1]
    H²τη_wξ_bot = m.H²τη_wξ[:, 1]

    # rhs τ
    τξ_rhs = τξ₀ - (τξ₀.*H²τξ_wξ_bot - τη₀.*H²τη_wξ_bot + H²τξ_b_bot)./m.H.^2
    τη_rhs = τη₀ - (τξ₀.*H²τη_wξ_bot + τη₀.*H²τξ_wξ_bot + H²τη_b_bot)./m.H.^2

    # get barotropic_RHS
    barotropic_RHS = get_barotropic_RHS(m, γ, τξ_rhs, τη_rhs)

    # solve
    Ψ = m.barotropic_LHS\barotropic_RHS
    println("Ψ: ", maximum(abs.(Ψ/1e6)))

    # get H²τ
    H²τξ, H²τη = get_full_τ(m, H²τξ_b, H²τη_b, τξ₀, τη₀, Ψ)

    # convert to Huξ, Huη, Huσ
    Huξ, Huη, Huσ = get_u(m, H²τξ, H²τη)

    if plots
        plot_horizontal(m.p, m.t, H²τξ_b_bot; clabel=L"Buoyancy bottom stress $H^2 \tau^\xi_b$ (kg m$^{-1}$ s$^{-2}$)", contours=false)
        savefig("images/tau_xi_b.png")
        println("images/tau_xi_b.png")
        plt.close()
        plot_horizontal(m.p, m.t, H²τη_b_bot; clabel=L"Buoyancy bottom stress $H^2 \tau^\eta_b$ (kg m$^{-1}$ s$^{-2}$)", contours=false)
        savefig("images/tau_eta_b.png")
        println("images/tau_eta_b.png")
        plt.close()
        plot_horizontal(m.p, m.t, Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)")
        savefig("images/psi.png")
        println("images/psi.png")
        plt.close()
    end

    return Ψ, Huξ, Huη, Huσ
end
# function invert!(m::ModelSetup3DPG, s::ModelState3DPG)
#     Ψ, Huξ, Huη, Huσ = invert(m, τξ₀, τη₀, b)
#     s.Ψ[:] = Ψ
#     s.Huξ[:, :] = Huξ
#     s.Huη[:, :] = Huη
#     s.Huσ[:, :] = Huσ
# end