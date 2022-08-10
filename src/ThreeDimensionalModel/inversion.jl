function get_barotropic_LHS(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, e::AbstractArray{<:Integer,1},
    C₀::AbstractArray{<:Real,3}, ρ₀::Real, f₀::Real, β::Real, H::AbstractArray{<:Real,1}, 
    Hx::AbstractArray{<:Real,1}, Hy::AbstractArray{<:Real,1}, τξ_tξ_bot::AbstractArray{<:Real,1}, 
    τη_tξ_bot::AbstractArray{<:Real,1})
    # indices
    np = size(p, 1)
    nt = size(t, 1)
    ne = size(e, 1)

    # number of shape functions per triangle
    n = size(t, 2)

    # α = τ/ρ₀/H
    αξ_tξ_bot = @. τξ_tξ_bot/ρ₀/H
    αη_tξ_bot = @. τη_tξ_bot/ρ₀/H
    
    # f/H
    f_over_H = (f₀ .+ β*p[:, 2])./H

    # functions
    H_func(ξ, η, k)         = fem_evaluate(H,         ξ, η, p, t, C₀, k)
    Hx_func(ξ, η, k)        = fem_evaluate(Hx,        ξ, η, p, t, C₀, k)
    Hy_func(ξ, η, k)        = fem_evaluate(Hy,        ξ, η, p, t, C₀, k)
    αξ_tξ_bot_func(ξ, η, k) = fem_evaluate(αξ_tξ_bot, ξ, η, p, t, C₀, k)
    αη_tξ_bot_func(ξ, η, k) = fem_evaluate(αη_tξ_bot, ξ, η, p, t, C₀, k)

    τξ_tξ_bot_func(ξ, η, k) = fem_evaluate(τξ_tξ_bot, ξ, η, p, t, C₀, k)
    τη_tξ_bot_func(ξ, η, k) = fem_evaluate(τη_tξ_bot, ξ, η, p, t, C₀, k)

    # create global linear system using stamping method
    barotropic_LHS = Tuple{Int64,Int64,Float64}[]
    @showprogress "Building barotropic_LHS..." for k = 1:nt
        # calculate contribution to K from element k
        Kᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                # func(ξ, η) = -αξ_tξ_bot_func(ξ, η, k)*
                func(ξ, η) = -τξ_tξ_bot_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)*
                             (shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η; dξ=1) + 
                              shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dη=1))
                Kᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to K′ from element k
        K′ᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                # func(ξ, η) = αη_tξ_bot_func(ξ, η, k)*
                func(ξ, η) = τη_tξ_bot_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)*
                             (shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dξ=1) - 
                              shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η; dη=1))
                K′ᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to C from element k
        Cᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                # func(ξ, η) = ∂η(f_over_H, ξ, η, k, p, t, C₀)*shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η) -
                #              ∂ξ(f_over_H, ξ, η, k, p, t, C₀)*shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η)
                 func(ξ, η) = (β/H_func(ξ, η, k) - (f₀ + β*η)*Hy_func(ξ, η, k)/H_func(ξ, η, k)^2)*
                             shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η) -
                             -(f₀ + β*η)*Hx_func(ξ, η, k)/H_func(ξ, η, k)^2*
                             shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η)
                Cᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # add to global system
        for i=1:n
            for j=1:n
                if t[k, i] in e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(barotropic_LHS, (t[k, i], t[k, j], Kᵏ[i, j]))
                push!(barotropic_LHS, (t[k, i], t[k, j], K′ᵏ[i, j]))
                push!(barotropic_LHS, (t[k, i], t[k, j], Cᵏ[i, j]))
            end
        end
    end
    # dirichlet Ψ = 0 along edges
    for i=1:ne
        push!(barotropic_LHS, (e[i], e[i], 1))
    end

    # make CSC matrix
    barotropic_LHS = sparse((x -> x[1]).(barotropic_LHS), (x -> x[2]).(barotropic_LHS), (x -> x[3]).(barotropic_LHS), np, np)

    return lu(barotropic_LHS)
end

function get_barotropic_RHS(m::ModelSetup3DPG, γ::AbstractArray{<:Real,1}, τξ::AbstractArray{<:Real,1},
                            τη::AbstractArray{<:Real,1})
    # number of shape functions per triangle
    n = size(m.t, 2)

    # functions
    H_func(ξ, η, k)  = fem_evaluate(m, m.H,  ξ, η, k)
    Hx_func(ξ, η, k) = fem_evaluate(m, m.Hx, ξ, η, k)
    Hy_func(ξ, η, k) = fem_evaluate(m, m.Hy, ξ, η, k)
    τξ_func(ξ, η, k) = fem_evaluate(m, τξ,   ξ, η, k)
    τη_func(ξ, η, k) = fem_evaluate(m, τη,   ξ, η, k)
    curl_τ(ξ, η, k)  = ∂ξ(m, τη, ξ, η, k)/H_func(ξ, η, k) - τη_func(ξ, η, k)/H_func(ξ, η, k)^2*Hx_func(ξ, η, k) -
                      (∂η(m, τξ, ξ, η, k)/H_func(ξ, η, k) - τξ_func(ξ, η, k)/H_func(ξ, η, k)^2*Hy_func(ξ, η, k))
    JEBAR(ξ, η, k)   = 1/H_func(ξ, η, k)^2 * (Hx_func(ξ, η, k)*∂η(m, γ, ξ, η, k) - Hy_func(ξ, η, k)*∂ξ(m, γ, ξ, η, k))

    # # compute curl of τ/ρ₀/H
    # αξ = @. τξ/m.ρ₀/m.H
    # αη = @. τη/m.ρ₀/m.H
    # barotropic_RHS = m.Cξ*αη .- m.Cη*αξ
    barotropic_RHS = zeros(m.np)

	# stamp JEBAR
	@showprogress "Building barotropic_RHS..." for k=1:m.nt
        for i=1:n
            if m.t[k, i] in m.e
                # edge node, leave as zero so that Ψ = 0
                continue
            end
            # func(ξ, η) = -JEBAR(ξ, η, k)*shape_func(m.C₀[k, i, :], ξ, η)
            func(ξ, η) = (-JEBAR(ξ, η, k) + curl_τ(ξ, η, k)/m.ρ₀)*shape_func(m.C₀[k, i, :], ξ, η)
            barotropic_RHS[m.t[k, i]] += tri_quad(func, m.p[m.t[k, 1:3], :]; degree=4)
        end
	end

    return barotropic_RHS
end

function get_m(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # indices
	np = size(p, 1)
	nt = size(t, 1)

    # number of shape functions per triangle
    n = size(t, 2)

	# create global linear system using stamping method
    m = zeros(np)
	for k=1:nt
		# add contribution to m from element k
        for i=1:n
            func(ξ, η) = shape_func(C₀[k, i, :], ξ, η)
            m[t[k, i]] += tri_quad(func, p[t[k, 1:3], :]; degree=4)
        end
	end

    return m
end

function get_baroclinic_LHS(ρ₀::Real, ν::AbstractArray{<:Real,1}, f::Real, H::Real, σ::AbstractArray{<:Real,1})
    # convention: τξ is variable 1, τη is variable 2
    nσ = size(σ, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_LHS = Tuple{Int64,Int64,Float64}[]  

    # Interior nodes
    @inbounds for j=2:nσ-1 
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = rhs_x
        row = imap[1, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[1, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(baroclinic_LHS, (row, imap[1, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(baroclinic_LHS, (row, imap[1, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(baroclinic_LHS, (row, imap[2, j], 1))

        # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = rhs_y
        row = imap[2, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[2, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(baroclinic_LHS, (row, imap[2, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(baroclinic_LHS, (row, imap[2, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(baroclinic_LHS, (row, imap[1, j], -1))
    end

    # Upper boundary conditions: wind stress
    # b.c. 1: τξ = τξ₀ at σ = 0
    push!(baroclinic_LHS, (imap[1, nσ], imap[1, nσ], 1))
    # b.c. 2: τη = τη₀ at σ = 0
    push!(baroclinic_LHS, (imap[2, nσ], imap[2, nσ], 1))

    # Integral boundary conditions: transport
    # b.c. 1: -H² ∫ σ τξ/ρ₀/ν dσ = Uξ
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end
    # b.c. 1: -H² ∫ σ τη/ρ₀/ν dσ = Uη
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[2, 1], imap[2, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[2, 1], imap[2, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    baroclinic_LHS = sparse((x->x[1]).(baroclinic_LHS), (x->x[2]).(baroclinic_LHS), (x->x[3]).(baroclinic_LHS), nvar*nσ, nvar*nσ)

    return lu(baroclinic_LHS)
end

function get_baroclinic_RHS(rhs_x::AbstractArray{<:Real,1}, rhs_y::AbstractArray{<:Real,1}, 
                            τξ₀::Real, τη₀::Real, Uξ::Real, Uη::Real)
    nσ = size(rhs_x, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_RHS = zeros(nvar*nσ)

    # eqtns:
    # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = rhs_x
    baroclinic_RHS[imap[1, 2:nσ-1]] = rhs_x[2:nσ-1] 
    # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = rhs_y
    baroclinic_RHS[imap[2, 2:nσ-1]] = rhs_y[2:nσ-1] 

    # top b.c.:
    # b.c. 1: τξ = τξ₀ at σ = 0
    baroclinic_RHS[imap[1, nσ]] = τξ₀
    # b.c. 2: τη = τη₀ at σ = 0
    baroclinic_RHS[imap[2, nσ]] = τη₀
    
    # integral b.c.:
    # b.c. 1: -H² ∫ σ τξ/ρ₀/ν dσ = Uξ
    baroclinic_RHS[imap[1, 1]] = Uξ
    # b.c. 2: -H² ∫ σ τη/ρ₀/ν dσ = Uη
    baroclinic_RHS[imap[2, 1]] = Uη

    return baroclinic_RHS
end

function solve_baroclinic_systems(baroclinic_LHSs::AbstractArray{SuiteSparse.UMFPACK.UmfpackLU,1}, 
                                  baroclinic_RHSs::AbstractArray{<:Real,2})
    np = size(baroclinic_RHSs, 1)
    nσ = Int64(size(baroclinic_RHSs, 2)/2)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    τξ = zeros(np, nσ)
    τη = zeros(np, nσ)
    for i=1:np
        sol = baroclinic_LHSs[i]\baroclinic_RHSs[i, :]
        τξ[i, :] = sol[imap[1, :]]
        τη[i, :] = sol[imap[2, :]]
    end
    return τξ, τη
end

"""
    τξ_b, τη_b = get_τ_b(m, b)
"""
function get_τ_b(m::ModelSetup3DPG, b::AbstractArray{<:Real,2})
    # rhs_x = zeros(m.np, m.nσ)
    # rhs_y = zeros(m.np, m.nσ)
    # N² = m.N²[1, 1]
    # for j=1:m.nσ
    #     bξ = m.Hx./m.H.*b[:, j]
    #     bη = m.Hy./m.H.*b[:, j]
    #     bσ = N²*m.H*(1 - exp(-(m.σ[j] + 1)/0.1))
    #     bx = bξ - m.σ[j]*m.Hx./m.H.*bσ
    #     by = bη - m.σ[j]*m.Hy./m.H.*bσ
    #     rhs_x[:, j] = m.ρ₀*m.ν[:, j]./(m.f₀ .+ m.β*m.p[:, 2]).*bx
    #     rhs_y[:, j] = m.ρ₀*m.ν[:, j]./(m.f₀ .+ m.β*m.p[:, 2]).*by
    # end
    # baroclinic_RHSs_b = zeros(m.np, 2*m.nσ)
    # for i=1:m.np
    #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
    # end
    # τξ_b, τη_b = solve_baroclinic_systems(m.baroclinic_LHSs, baroclinic_RHSs_b)

    # # integrals of buoyancy gradients on rhs
    # bσ_x = zeros(np, m.nσ)
    # bσ_y = zeros(np, m.nσ)
    # for i=1:np
    #     bσ_x[i, :] = -m.σ*Hx[i]/H[i].*differentiate(b[i, :], m.σ) 
    #     bσ_y[i, :] = -m.σ*Hy[i]/H[i].*differentiate(b[i, :], m.σ)
    # end
    # rhs_x = m.Cξ*b + m.M*bσ_x
    # rhs_y = m.Cη*b + m.M*bσ_y
    # for i=1:np
    #     rhs_x[i, :] .*= m.ρ₀*m.ν[i, :]/(m.f₀ + m.β*η[i])
    #     rhs_y[i, :] .*= m.ρ₀*m.ν[i, :]/(m.f₀ + m.β*η[i])
    # end
    # # stress due to buoyancy gradients
    # baroclinic_RHSs_b = zeros(np, 2*m.nσ)
    # for i=1:np
    #     baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(rhs_x[i, :], rhs_y[i, :], 0, 0, 0, 0)
    # end
    # vξ_b, vη_b = solve_baroclinic_systems(m.baroclinic_LHSs, baroclinic_RHSs_b)
    # τξ_b = m.M_LU\vξ_b
    # τη_b = m.M_LU\vη_b

    # pointwise buoyancy gradients
    b_x = m.M_LU\(m.Cξ*b)
    b_y = m.M_LU\(m.Cη*b)
    for i=1:m.np
        b_x[i, :] += -m.σ*m.Hx[i].*differentiate(b[i, :], m.σ)/m.H[i] 
        b_y[i, :] += -m.σ*m.Hy[i].*differentiate(b[i, :], m.σ)/m.H[i]
    end
    # stress due to buoyancy gradients
    baroclinic_RHSs_b = zeros(m.np, 2*m.nσ)
    for i=1:m.np
        coeff = m.ρ₀*m.ν[i, :]./(m.f₀ .+ m.β*m.p[i, 2])
        baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(coeff.*b_x[i, :], coeff.*b_y[i, :], 0, 0, 0, 0)
    end
    τξ_b, τη_b = solve_baroclinic_systems(m.baroclinic_LHSs, baroclinic_RHSs_b)

    return τξ_b, τη_b
end

"""
    τξ, τη = get_full_τ(m, τξ_b, τη_b, τξ₀, τη₀, Ψ)
"""
function get_full_τ(m, τξ_b, τη_b, τξ₀, τη₀, Ψ)
    # vξ = zeros(m.np, m.nσ)
    # vη = zeros(m.np, m.nσ)
    # n = size(m.t, 2)
    # @showprogress "Computing full τ (assuming τ₀ = 0)..." for k=1:m.nt
    #     # precompute matrix mult on Ψ
    #     Aξ = reshape(reshape(m.CCξ[k, :, :, :], n^2, n)*Ψ[m.t[k, :]], n, n)
    #     Aη = reshape(reshape(m.CCη[k, :, :, :], n^2, n)*Ψ[m.t[k, :]], n, n)
    #     for j=1:m.nσ
    #         # now mult τ
    #         vξ[m.t[k, :], j] += -Aη*m.τξ_tξ[m.t[k, :], j] - Aξ*m.τη_tξ[m.t[k, :], j]
    #         vη[m.t[k, :], j] += -Aη*m.τη_tξ[m.t[k, :], j] + Aξ*m.τξ_tξ[m.t[k, :], j]
    #     end
    # end
    # τξ = τξ_b + m.M_LU\vξ
    # τη = τη_b + m.M_LU\vη

    Uξ = -(m.M_LU\(m.Cη*Ψ))
    Uη =  m.M_LU\(m.Cξ*Ψ)
    # plot_horizontal(m.p, m.t, Uξ)
    # savefig("images/Uxi.png")
    # plt.close()
    # plot_horizontal(m.p, m.t, Uη)
    # savefig("images/Ueta.png")
    # plt.close()
    τξ = @. τξ_b #+ Uξ*m.τξ_tξ - Uη*m.τη_tξ
    τη = @. τη_b #+ Uξ*m.τη_tξ + Uη*m.τξ_tξ
    return τξ, τη
end

"""
    uξ, uη, uσ = get_u(m, τξ, τη)
"""
function get_u(m::ModelSetup3DPG, τξ::AbstractArray{<:Real,2}, τη::AbstractArray{<:Real,2})
    # integrate τξ and τη to get uξ and uη
    uξ = zeros(m.np, m.nσ)
    uη = zeros(m.np, m.nσ)
    for i=1:m.np
        uξ[i, :] = m.H[i]/m.ρ₀*cumtrapz(τξ[i, :]./m.ν[i, :], m.σ)
        uη[i, :] = m.H[i]/m.ρ₀*cumtrapz(τη[i, :]./m.ν[i, :], m.σ)
    end

    # integrate divergence of uξ and uη to get uσ
    # div = m.M_LU\(m.Cξ*(m.H.*uξ) + m.Cη*(m.H.*uη))
    # uσ = zeros(m.np, m.nσ)
    # for i=1:m.np
    #     uσ[i, :] = 1/m.H[i]*cumtrapz(-div[i, :], m.σ)
    # end
    Dσσ = get_Dσσ(m.σ)
    rhs = -(m.M_LU\(m.Cξ*(m.H.^2/m.ρ₀./m.ν.*τξ) + m.Cη*(m.H.^2/m.ρ₀./m.ν.*τη)))
    rhs[:, 1] .= 0
    rhs[:, m.nσ] .= 0
    uσ = zeros(m.np, m.nσ)
    for i=1:m.np
        uσ[i, :] = 1/m.H[i]*(Dσσ\rhs[i, :])
    end

    return uξ, uη, uσ
end

"""
    Ψ, uξ, uη, uσ = invert(m, τξ₀, τη₀, b)
"""
function invert(m::ModelSetup3DPG, τξ₀::AbstractArray{<:Real,1}, τη₀::AbstractArray{<:Real,1}, 
                b::AbstractArray{<:Real,2}; plots=false)
    # solve for stress due to buoyancy gradients
    τξ_b, τη_b = get_τ_b(m, b)

    # bottom stress 
    τξ_b_bot = τξ_b[:, 1]
    τη_b_bot = τη_b[:, 1]

    # buoyancy integral for JEBAR term
    γ = zeros(m.np)
    for i=1:m.np
        γ[i] = -m.H[i]^2*trapz(m.σ.*b[i, :], m.σ)
    end

    # bottom stress due to wind stress
    τξ_wξ_bot = m.τξ_wξ[:, 1]
    τη_wξ_bot = m.τη_wξ[:, 1]

    # rhs τ
    τξ_rhs = τξ₀ - (τξ₀.*τξ_wξ_bot - τη₀.*τη_wξ_bot) - τξ_b_bot
    τη_rhs = τη₀ - (τξ₀.*τη_wξ_bot + τη₀.*τξ_wξ_bot) - τη_b_bot

    # get barotropic_RHS
    barotropic_RHS = get_barotropic_RHS(m, γ, τξ_rhs, τη_rhs)

    # solve
    Ψ = m.barotropic_LHS\barotropic_RHS

    # get τ
    τξ, τη = get_full_τ(m, τξ_b, τη_b, τξ₀, τη₀, Ψ)

    # convert to uξ, uη, uσ
    uξ, uη, uσ = get_u(m, τξ, τη)

    if plots
        plot_horizontal(m.p, m.t, τξ_b_bot; clabel=L"Buoyancy bottom stress $\tau^\xi_b$ (kg m$^{-1}$ s$^{-2}$)")
        savefig("images/tau_xi_b.png")
        println("images/tau_xi_b.png")
        plt.close()
        plot_horizontal(m.p, m.t, τη_b_bot; clabel=L"Buoyancy bottom stress $\tau^\eta_b$ (kg m$^{-1}$ s$^{-2}$)")
        savefig("images/tau_eta_b.png")
        println("images/tau_eta_b.png")
        plt.close()
        plot_horizontal(m.p, m.t, γ; clabel=L"Buoyancy integral $\gamma$ (m$^{3}$ s$^{-2}$)")
        savefig("images/gamma.png")
        println("images/gamma.png")
        plt.close()
        fig, ax, im = plot_horizontal(m.p, m.t, Ψ/1e6; clabel=L"Streamfunction $\Psi$ (Sv)")
        savefig("images/psi.png")
        println("images/psi.png")
        plt.close()
    end

    return Ψ, uξ, uη, uσ
end
# function invert!(m::ModelSetup3DPG, s::ModelState3DPG)
#     Ψ, uξ, uη, uσ = invert(m, τξ₀, τη₀, b)
#     s.Ψ[:] = Ψ
#     s.uξ[:, :] = uξ
#     s.uη[:, :] = uη
#     s.uσ[:, :] = uσ
# end