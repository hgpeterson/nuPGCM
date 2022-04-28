function get_K(p, t, e, C₀, ρ₀, H, τξ_tξ_bot)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]  
	for k=1:n_tri
		# calculate K matrix element Kₑ
        Kₑ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = -τξ_tξ_bot(ξ, η)/ρ₀/H(ξ, η)*(C₀[k, 2, j]*C₀[k, 2, i] + C₀[k, 3, j]*C₀[k, 3, i])
                Kₑ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                if t[k, i] in e
                    # edge node, leave for E
                    continue
                end
                push!(K, (imap[t[k, i]], imap[t[k, j]], Kₑ[i, j]))
			end
		end
	end

    # make CSC matrix
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), n_nodes, n_nodes)

    return K
end

function get_K′(p, t, e, C₀, ρ₀, H, τη_tξ_bot)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    K′ = Tuple{Int64,Int64,Float64}[]  
	for k=1:n_tri
		# calculate K′ matrix element Kₑ′
        Kₑ′ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = τη_tξ_bot(ξ, η)/ρ₀/H(ξ, η)*(C₀[k, 3, j]*C₀[k, 2, i] - C₀[k, 2, j]*C₀[k, 3, i])
                Kₑ′[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                if t[k, i] in e
                    # edge node, leave for E
                    continue
                end
                push!(K′, (imap[t[k, i]], imap[t[k, j]], Kₑ′[i, j]))
			end
		end
	end

    # make CSC matrix
    K′ = sparse((x->x[1]).(K′), (x->x[2]).(K′), (x->x[3]).(K′), n_nodes, n_nodes)

    return K′
end

function get_C(p, t, e, C₀, f, fy, H, Hx, Hy)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    C = Tuple{Int64,Int64,Float64}[]  
	for k=1:n_tri
		# calculate C matrix elements Cₑ
        Cₑ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = (fy(ξ, η)/H(ξ, η) - f(ξ, η)*Hy(ξ, η)/H(ξ, η)^2)*C₀[k, 2, j]*local_basis_func(C₀[k, :, i], [ξ, η]) - 
                             -f(ξ, η)*Hx(ξ, η)/H(ξ, η)^2*C₀[k, 3, j]*local_basis_func(C₀[k, :, i], [ξ, η])
                Cₑ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                if t[k, i] in e
                    # edge node, leave for E
                    continue
                end
                push!(C, (imap[t[k, i]], imap[t[k, j]], Cₑ[i, j]))
			end
		end
	end

    # make CSC matrices
    C = sparse((x->x[1]).(C), (x->x[2]).(C), (x->x[3]).(C), n_nodes, n_nodes)

    return C
end

function get_E(p, e)
	n_nodes = size(p, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

    # dirichlet Ψ = 0 along edges
    E = Tuple{Int64,Int64,Float64}[]  
    for i=1:size(e, 1)
        push!(E, (imap[e[i]], imap[e[i]], 1))
    end
    
    # make CSC matrix
    E = sparse((x->x[1]).(E), (x->x[2]).(E), (x->x[3]).(E), n_nodes, n_nodes)

    return E
end

function get_barotropic_LHS(p, t, e, C₀, ρ₀, f, fy, H, Hx, Hy, τξ_tξ_bot, τη_tξ_bot)
    # build matrices
    println("building K")
    K = get_K(p, t, e, C₀, ρ₀, H, τξ_tξ_bot)
    println("building K′")
    K′ = get_K′(p, t, e, C₀, ρ₀, H, τη_tξ_bot)
    println("building C")
    C = get_C(p, t, e, C₀, f, fy, H, Hx, Hy)
    println("building E")
    E = get_E(p, e)

    # full barotropic_LHS matrix
    barotropic_LHS = K + K′ + C + E

    return lu(barotropic_LHS)
end

function get_barotropic_RHS(p, t, e, C₀, F)
    np = size(p, 1)
    nt = size(t, 1)
    imap = reshape(1:np, 1, np) 

	# create global linear system using stamping method
    barotropic_RHS = zeros(np)
	for k=1:nt
		# calculate barotropic_RHS vector element and add it to the global system
        for i=1:3
            if t[k, i] in e
                # edge node, leave as zero so that Ψ = 0
                continue
            end
            f(ξ, η) = F(ξ, η)*local_basis_func(C₀[k, :, i], [ξ, η])
            barotropic_RHS[imap[t[k, i]]] += gaussian_quad2(f, p[t[k, :], :])
        end
	end

    return barotropic_RHS
end
function get_barotropic_RHS(m::ModelSetup3DPG, F)
    return get_barotropic_RHS(m.p, m.t, m.e, m.C₀, F)
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

        # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = ν/ρ₀/f ∂x(b)
        row = imap[1, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[1, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(baroclinic_LHS, (row, imap[1, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(baroclinic_LHS, (row, imap[1, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(baroclinic_LHS, (row, imap[2, j], 1))

        # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = ν/ρ₀/f ∂y(b)
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
    # b.c. 1: -H² ∫ στξ/ρ₀/ν dσ = Uξ
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end
    # b.c. 1: -H² ∫ στη/ρ₀/ν dσ = Uη
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[2, 1], imap[2, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[2, 1], imap[2, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    baroclinic_LHS = sparse((x->x[1]).(baroclinic_LHS), (x->x[2]).(baroclinic_LHS), (x->x[3]).(baroclinic_LHS), nvar*nσ, nvar*nσ)

    return lu(baroclinic_LHS)
end

function get_baroclinic_RHS(rhs_x::AbstractArray{<:Real,1}, rhs_y::AbstractArray{<:Real,1}, τξ₀::Real, τη₀::Real, Uξ::Real, Uη::Real)
    nσ = size(rhs_x, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_RHS = zeros(nvar*nσ)

    # eqtns:
    # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = ν/ρ₀/f ∂x(b)
    baroclinic_RHS[imap[1, 2:nσ-1]] = rhs_x[2:nσ-1] 
    # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = ν/ρ₀/f ∂y(b)
    baroclinic_RHS[imap[2, 2:nσ-1]] = rhs_y[2:nσ-1] 

    # top b.c.:
    # b.c. 1: τξ = τξ₀ at σ = 0
    baroclinic_RHS[imap[1, nσ]] = τξ₀
    # b.c. 2: τη = τη₀ at σ = 0
    baroclinic_RHS[imap[2, nσ]] = τη₀
    
    # integral b.c.:
    # b.c. 1: -H² ∫ στξ/ρ₀/ν dσ = Uξ
    baroclinic_RHS[imap[1, 1]] = Uξ
    # b.c. 2: -H² ∫ στη/ρ₀/ν dσ = Uη
    baroclinic_RHS[imap[2, 1]] = Uη

    return baroclinic_RHS
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
