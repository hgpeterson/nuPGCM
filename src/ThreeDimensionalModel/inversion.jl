function get_K(p, t, e, C₀, ρ₀, H, τ_tξ)
    # indices
	np = size(p, 1)
	nt = size(t, 1)

    # functions
    H_func(ξ, η, k) = evaluate(H, [ξ, η], p, t, C₀, k)
    τξ_tξ_bot_func(ξ, η, k) = evaluate(τ_tξ[1, :, 1], [ξ, η], p, t, C₀, k)

	# create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]  
	@showprogress "Building K..." for k=1:nt
		# calculate contribution to K from element k
        Kᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = -τξ_tξ_bot_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)*(C₀[k, 2, j]*C₀[k, 2, i] + C₀[k, 3, j]*C₀[k, 3, i])
                Kᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                if t[k, i] in e
                    # edge node, leave for E
                    continue
                end
                push!(K, (t[k, i], t[k, j], Kᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), np, np)

    return K
end

function get_K′(p, t, e, C₀, ρ₀, H, τ_tξ)
    # indices
	np = size(p, 1)
	nt = size(t, 1)

    # functions
    H_func(ξ, η, k) = evaluate(H, [ξ, η], p, t, C₀, k)
    τη_tξ_bot_func(ξ, η, k) = evaluate(τ_tξ[2, :, 1], [ξ, η], p, t, C₀, k)

	# create global linear system using stamping method
    K′ = Tuple{Int64,Int64,Float64}[]  
	@showprogress "Building K′..." for k=1:nt
		# calculate contribution to K′ from element k
        K′ᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = τη_tξ_bot_func(ξ, η, k)/ρ₀/H_func(ξ, η, k)*(C₀[k, 3, j]*C₀[k, 2, i] - C₀[k, 2, j]*C₀[k, 3, i])
                K′ᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                if t[k, i] in e
                    # edge node, leave for E
                    continue
                end
                push!(K′, (t[k, i], t[k, j], K′ᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    K′ = sparse((x->x[1]).(K′), (x->x[2]).(K′), (x->x[3]).(K′), np, np)

    return K′
end

function get_C(p, t, e, C₀, f, fy, H, Hx, Hy)
    # indices
	np = size(p, 1)
	nt = size(t, 1)

    # functions
    H_func(ξ, η, k) = evaluate(H, [ξ, η], p, t, C₀, k)
    Hx_func(ξ, η, k) = evaluate(Hx, [ξ, η], p, t, C₀, k)
    Hy_func(ξ, η, k) = evaluate(Hy, [ξ, η], p, t, C₀, k)
    f_func(ξ, η, k) = evaluate(f, [ξ, η], p, t, C₀, k)
    fy_func(ξ, η, k) = evaluate(fy, [ξ, η], p, t, C₀, k)

	# create global linear system using stamping method
    C = Tuple{Int64,Int64,Float64}[]  
	@showprogress "Building C..." for k=1:nt
		# calculate contribution to C from element k
        Cᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = (fy_func(ξ, η, k)/H_func(ξ, η, k) - f_func(ξ, η, k)*Hy_func(ξ, η, k)/H_func(ξ, η, k)^2)*C₀[k, 2, j]*local_basis_func(C₀[k, :, i], [ξ, η]) - 
                             -f_func(ξ, η, k)*Hx_func(ξ, η, k)/H_func(ξ, η, k)^2*C₀[k, 3, j]*local_basis_func(C₀[k, :, i], [ξ, η])
                Cᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                if t[k, i] in e
                    # edge node, leave for E
                    continue
                end
                push!(C, (t[k, i], t[k, j], Cᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    C = sparse((x->x[1]).(C), (x->x[2]).(C), (x->x[3]).(C), np, np)

    return C
end

function get_E(p, e)
    # indices
	np = size(p, 1)
	ne = size(e, 1)

    # dirichlet Ψ = 0 along edges
    E = Tuple{Int64,Int64,Float64}[]  
    @showprogress "Building E..." for i=1:ne
        push!(E, (e[i], e[i], 1))
    end
    
    # make CSC matrix
    E = sparse((x->x[1]).(E), (x->x[2]).(E), (x->x[3]).(E), np, np)

    return E
end

function get_barotropic_LHS(p, t, e, C₀, ρ₀, f, fy, H, Hx, Hy, τ_tξ)
    # build matrices
    K = get_K(p, t, e, C₀, ρ₀, H, τ_tξ)
    K′ = get_K′(p, t, e, C₀, ρ₀, H, τ_tξ)
    C = get_C(p, t, e, C₀, f, fy, H, Hx, Hy)
    E = get_E(p, e)

    # full barotropic_LHS matrix
    barotropic_LHS = K + K′ + C + E

    return lu(barotropic_LHS)
end

function get_barotropic_RHS(m::ModelSetup3DPG, γ, τ)
    # functions
    JEBAR(ξ, η, k) = 0 # for now
    H_func(ξ, η, k) = evaluate(m.H, [ξ, η], m.p, m.t, m.C₀, k)
    τξ_func(ξ, η, k) = evaluate(τ[1, :], [ξ, η], m.p, m.t, m.C₀, k)
    τη_func(ξ, η, k) = evaluate(τ[2, :], [ξ, η], m.p, m.t, m.C₀, k)
    # curl of stress ∂ξ(τη/H) - ∂η(τξ/H)
    curl_τ(ξ, η, k) = ∂ξ(m, τ[2, :], [ξ, η], k)/H_func(ξ, η, k) - τη_func(ξ, η, k)/H_func(ξ, η, k)^2*∂ξ(m, m.H, [ξ, η], k) -
                      (∂η(m, τ[1, :], [ξ, η], k)/H_func(ξ, η, k) - τξ_func(ξ, η, k)/H_func(ξ, η, k)^2*∂η(m, m.H, [ξ, η], k))

	# create global linear system using stamping method
    barotropic_RHS = zeros(m.np)
	@showprogress "Building F..." for k=1:m.nt
		# calculate barotropic_RHS vector element and add it to the global system
        for i=1:3
            if m.t[k, i] in m.e
                # edge node, leave as zero so that Ψ = 0
                continue
            end
            f(ξ, η) = (JEBAR(ξ, η, k) + curl_τ(ξ, η, k)/m.ρ₀)*local_basis_func(m.C₀[k, :, i], [ξ, η])
            barotropic_RHS[m.t[k, i]] += gaussian_quad2(f, m.p[m.t[k, :], :])
        end
	end

    return barotropic_RHS
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

function get_baroclinic_RHS(rhs_x::AbstractArray{<:Real,1}, rhs_y::AbstractArray{<:Real,1}, vξ₀::Real, vη₀::Real, Uξφ::Real, Uηφ::Real)
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

function get_M(p, t, C₀)
    # indices
	np = size(p, 1)
	nt = size(t, 1)

	# create global linear system using stamping method
    M = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to M from element k
        Mᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = local_basis_func(C₀[k, :, j], [ξ, η])*local_basis_func(C₀[k, :, i], [ξ, η])
                Mᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                push!(M, (t[k, i], t[k, j], Mᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    M = sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), np, np)

    return M
end

function get_τ(baroclinic_LHSs::AbstractArray{Any,1}, baroclinic_RHSs::AbstractArray{<:Real,2})
    np = size(baroclinic_RHSs, 1)
    nσ = Int64(size(baroclinic_RHSs, 2)/2)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    τ = zeros(2, np, nσ)
    @inbounds for i=1:np
        if baroclinic_LHSs[i] === nothing
            continue
        else
            sol = baroclinic_LHSs[i]\baroclinic_RHSs[i, :]
            τ[1, i, :] = sol[imap[1, :]]
            τ[2, i, :] = sol[imap[2, :]]
        end
    end
    return τ
end

function get_u(τ, ρ₀, ν, H, σ)
    u = zeros(size(τ))
    u[1, :, :] = cumtrapz(H/ρ₀./ν.*τ[1, :, :], σ)
    u[2, :, :] = cumtrapz(H/ρ₀./ν.*τ[2, :, :], σ)
    return u
end
function get_u(m, τ)
   get_u(τ, m.ρ₀, m.ν, m.H, m.σ) 
end
