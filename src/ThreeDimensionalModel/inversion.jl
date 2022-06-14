function get_barotropic_LHS(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, e::AbstractArray{<:Integer,1},
    C₀::AbstractArray{<:Real,3}, ρ₀::Real, f₀::Real, β::Real, H::AbstractArray{<:Real,1}, 
    Hx::AbstractArray{<:Real,1}, Hy::AbstractArray{<:Real,1}, τξ_tξ_bot::AbstractArray{<:Real,1}, 
    τη_tξ_bot::AbstractArray{<:Real,1})
    # indices
    np = size(p, 1)
    nt = size(t, 1)
    ne = size(e, 1)

    # functions
    H_func(ξ, η, k)         = fem_evaluate(H,         ξ, η, p, t, C₀, k)
    Hx_func(ξ, η, k)        = fem_evaluate(Hx,        ξ, η, p, t, C₀, k)
    Hy_func(ξ, η, k)        = fem_evaluate(Hy,        ξ, η, p, t, C₀, k)
    τξ_tξ_bot_func(ξ, η, k) = fem_evaluate(τξ_tξ_bot, ξ, η, p, t, C₀, k)
    τη_tξ_bot_func(ξ, η, k) = fem_evaluate(τη_tξ_bot, ξ, η, p, t, C₀, k)

    # create global linear system using stamping method
    barotropic_LHS = Tuple{Int64,Int64,Float64}[]
    @showprogress "Building barotropic_LHS..." for k = 1:nt
        # calculate contribution to K from element k
        Kᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = -τξ_tξ_bot_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)*(C₀[k, 2, j]*C₀[k, 2, i] + C₀[k, 3, j]*C₀[k, 3, i])
                Kᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

        # calculate contribution to K′ from element k
        K′ᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = τη_tξ_bot_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)*(C₀[k, 3, j]*C₀[k, 2, i] - C₀[k, 2, j]*C₀[k, 3, i])
                K′ᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

        # calculate contribution to C from element k
        Cᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = (β/H_func(ξ, η, k) - (f₀ + β*η)*Hy_func(ξ, η, k)/H_func(ξ, η, k)^2)*C₀[k, 2, j]*shape_func(C₀[k, :, i], ξ, η) -
                             -(f₀ + β*η)*Hx_func(ξ, η, k)/H_func(ξ, η, k)^2*C₀[k, 3, j]*shape_func(C₀[k, :, i], ξ, η)
                Cᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

        # add to global system
        for i=1:3
            for j=1:3
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
    # functions
    H_func(ξ, η, k)  = fem_evaluate(m, m.H,  ξ, η, k)
    Hx_func(ξ, η, k) = fem_evaluate(m, m.Hx, ξ, η, k)
    Hy_func(ξ, η, k) = fem_evaluate(m, m.Hy, ξ, η, k)
    τξ_func(ξ, η, k) = fem_evaluate(m, τξ,   ξ, η, k)
    τη_func(ξ, η, k) = fem_evaluate(m, τη,   ξ, η, k)
    JEBAR(ξ, η, k)   = 1/H_func(ξ, η, k)^2 * (Hx_func(ξ, η, k)*∂η(m, γ, ξ, η, k) - Hy_func(ξ, η, k)*∂ξ(m, γ, ξ, η, k))
    curl_τ(ξ, η, k)  = ∂ξ(m, τη, ξ, η, k)/H_func(ξ, η, k) - τη_func(ξ, η, k)/H_func(ξ, η, k)^2*Hx_func(ξ, η, k) -
                      (∂η(m, τξ, ξ, η, k)/H_func(ξ, η, k) - τξ_func(ξ, η, k)/H_func(ξ, η, k)^2*Hy_func(ξ, η, k))

	# create global linear system using stamping method
    barotropic_RHS = zeros(m.np)
	@showprogress "Building barotropic_RHS..." for k=1:m.nt
	# for k=1:m.nt
		# calculate barotropic_RHS vector element and add it to the global system
        for i=1:3
            if m.t[k, i] in m.e
                # edge node, leave as zero so that Ψ = 0
                continue
            end
            func(ξ, η) = (JEBAR(ξ, η, k) + curl_τ(ξ, η, k)/m.ρ₀)*shape_func(m.C₀[k, :, i], ξ, η)
            barotropic_RHS[m.t[k, i]] += gaussian_quad2(func, m.p[m.t[k, :], :])
        end
	end

    return barotropic_RHS
end

function get_m(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # indices
	np = size(p, 1)
	nt = size(t, 1)

	# create global linear system using stamping method
    m = zeros(np)
	for k=1:nt
		# add contribution to m from element k
        for i=1:3
            func(ξ, η) = shape_func(C₀[k, :, i], ξ, η)
            m[t[k, i]] += gaussian_quad2(func, p[t[k, :], :])
        end
	end

    return m
end

function get_baroclinic_LHS(ρ₀::Real, ν::AbstractArray{<:Real,1}, f::Real, H::Real, σ::AbstractArray{<:Real,1})
    # convention: vξ is variable 1, vη is variable 2
    nσ = size(σ, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_LHS = Tuple{Int64,Int64,Float64}[]  

    # Interior nodes
    @inbounds for j=2:nσ-1 
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # eqtn 1: ν/f/H² ∂σσ(vξ) + vη = bˣ
        row = imap[1, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[1, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(baroclinic_LHS, (row, imap[1, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(baroclinic_LHS, (row, imap[1, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(baroclinic_LHS, (row, imap[2, j], 1))

        # eqtn 2: ν/f/H² ∂σσ(vη) - vξ = bʸ
        row = imap[2, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[2, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(baroclinic_LHS, (row, imap[2, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(baroclinic_LHS, (row, imap[2, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(baroclinic_LHS, (row, imap[1, j], -1))
    end

    # Upper boundary conditions: wind stress
    # b.c. 1: vξ = vξ₀ at σ = 0
    push!(baroclinic_LHS, (imap[1, nσ], imap[1, nσ], 1))
    # b.c. 2: vη = vη₀ at σ = 0
    push!(baroclinic_LHS, (imap[2, nσ], imap[2, nσ], 1))

    # Integral boundary conditions: transport
    # b.c. 1: -H² ∫ σ vξ/ρ₀/ν dσ = Uξφ
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end
    # b.c. 1: -H² ∫ σ vη/ρ₀/ν dσ = Uηφ
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
                            vξ₀::Real, vη₀::Real, Uξφ::Real, Uηφ::Real)
    nσ = size(rhs_x, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_RHS = zeros(nvar*nσ)

    # eqtns:
    # eqtn 1: ν/f/H² ∂σσ(vξ) + vη = bˣ
    baroclinic_RHS[imap[1, 2:nσ-1]] = rhs_x[2:nσ-1] 
    # eqtn 2: ν/f/H² ∂σσ(vη) - vξ = bʸ
    baroclinic_RHS[imap[2, 2:nσ-1]] = rhs_y[2:nσ-1] 

    # top b.c.:
    # b.c. 1: vξ = vξ₀ at σ = 0
    baroclinic_RHS[imap[1, nσ]] = vξ₀
    # b.c. 2: vη = vη₀ at σ = 0
    baroclinic_RHS[imap[2, nσ]] = vη₀
    
    # integral b.c.:
    # b.c. 1: -H² ∫ σ vξ/ρ₀/ν dσ = Uξφ
    baroclinic_RHS[imap[1, 1]] = Uξφ
    # b.c. 2: -H² ∫ σ vη/ρ₀/ν dσ = Uηφ
    baroclinic_RHS[imap[2, 1]] = Uηφ

    return baroclinic_RHS
end

function get_vξ_vη(baroclinic_LHSs::AbstractArray{SuiteSparse.UMFPACK.UmfpackLU,1}, baroclinic_RHSs::AbstractArray{<:Real,2})
    np = size(baroclinic_RHSs, 1)
    nσ = Int64(size(baroclinic_RHSs, 2)/2)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    vξ = zeros(np, nσ)
    vη = zeros(np, nσ)
    @showprogress "Calculating vξ and vη..." for i=1:np
        if baroclinic_LHSs[i] === nothing
            continue
        else
            sol = baroclinic_LHSs[i]\baroclinic_RHSs[i, :]
            vξ[i, :] = sol[imap[1, :]]
            vη[i, :] = sol[imap[2, :]]
        end
    end
    return vξ, vη
end

function get_uξ_uη(τξ, τη, ρ₀, ν, H, σ)
    uξ = zeros(size(τξ))
    uη = zeros(size(τξ))
    for i=1:size(uξ, 1)
        uξ[i, :] = cumtrapz(H[i]/ρ₀./ν[i, :].*τξ[i, :], σ)
        uη[i, :] = cumtrapz(H[i]/ρ₀./ν[i, :].*τη[i, :], σ)
    end
    return uξ, uη
end
function get_uξ_uη(m, τξ, τη)
   get_uξ_uη(τξ, τη, m.ρ₀, m.ν, m.H, m.σ) 
end
