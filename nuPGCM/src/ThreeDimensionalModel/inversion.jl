function get_K(p, t, e, H, τξ_t_bottom, τη_t_bottom)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]  
	for k=1:n_tri
		# get coeffs for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
        C₀ = get_linear_basis_coeffs(p[t[k, :], :])

		# calculate K matrix element Kₑ
        Kₑ = zeros(3, 3)
        for i=1:3
            for j=1:3
                f(ξ, η) = -τη_t_bottom(ξ, η)/H(ξ, η)*C₀[2, j]*C₀[2, i] + 
                          -τξ_t_bottom(ξ, η)/H(ξ, η)*C₀[3, j]*C₀[3, i]
                Kₑ[i, j] = gaussian_quad2(f, p[t[k, :], :])
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

function get_C(p, t, e, f₀, β, H, Hx, Hy)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    C = Tuple{Int64,Int64,Float64}[]  
	for k=1:n_tri
		# get coeffs for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
        C₀ = get_linear_basis_coeffs(p[t[k, :], :])

		# calculate C matrix elements Cₑ
        Cₑ = zeros(3, 3)
        for i=1:3
            for j=1:3
                f(ξ, η) = (β/H(ξ, η) - (f₀ + β*η)*Hy(ξ, η)/H(ξ, η)^2)*C₀[2, j]*local_basis_func(C₀[:, i], [ξ, η]) - 
                         -(f₀ + β*η)*Hx(ξ, η)/H(ξ, η)^2*C₀[3, j]*local_basis_func(C₀[:, i], [ξ, η])
                # f(p₀) = β/H(p₀[1], p₀[2])*C₀[2, j]*local_basis_func(C₀[:, i], p₀)
                Cₑ[i, j] = gaussian_quad2(f, p[t[k, :], :])
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

function get_barotropic_LHS(p, t, e, f₀, β, H, Hx, Hy, τξ_t_bottom, τη_t_bottom)
    # build matrices
    K = get_K(p, t, e, H, τξ_t_bottom, τη_t_bottom)
    C = get_C(p, t, e, f₀, β, H, Hx, Hy)
    E = get_E(p, e)

    # full barotropic_LHS matrix
    barotropic_LHS = C + K + E

    return lu(barotropic_LHS)
end
function get_barotropic_LHS(p, t, e, H, τξ_t_bottom, τη_t_bottom, C, E)
    # build K matrix
    K = get_K(p, t, e, H, τξ_t_bottom, τη_t_bottom)

    # full barotropic_LHS matrix
    barotropic_LHS = C + K + E

    return lu(barotropic_LHS)
end


function get_barotropic_RHS(p, t, e, F)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    barotropic_RHS = zeros(n_nodes)
	for k=1:n_tri
		# get coeffs for linear basis function c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
        C₀ = get_linear_basis_coeffs(p[t[k, :], :])

		# calculate barotropic_RHS vector element and add it to the global system
        for i=1:3
            if t[k, i] in e
                # edge node, leave as zero so that Ψ = 0
                continue
            end
            f(ξ, η) = F(ξ, η)*local_basis_func(C₀[:, i], [ξ, η])
            barotropic_RHS[imap[t[k, i]]] += gaussian_quad2(f, p[t[k, :], :])
        end
	end

    return barotropic_RHS
end

function get_baroclinic_LHS(ν::Vector{Float64}, f::Float64, H::Float64, σ::Vector{Float64})
    # convention: τξ is variable 1, τη is variable 2
    nσ = size(σ, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_LHS = Tuple{Int64,Int64,Float64}[]  

    # Interior nodes
    for j=2:nσ-1
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # eqtn 1: ∂σσ(τξ)/H² + f*τη/ν = 0 or ∂x(b)
        row = imap[1, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[1, j-1], fd_σσ[1]/H^2))
        push!(baroclinic_LHS, (row, imap[1, j],   fd_σσ[2]/H^2))
        push!(baroclinic_LHS, (row, imap[1, j+1], fd_σσ[3]/H^2))
        # term 2
        push!(baroclinic_LHS, (row, imap[2, j], f/ν[j]))

        # eqtn 2: ∂σσ(τη)/H² - f*τξ/ν = 0 or ∂y(b)
        row = imap[1, j]
        # term 1
        push!(baroclinic_LHS, (row, imap[2, j-1], fd_σσ[1]/H^2))
        push!(baroclinic_LHS, (row, imap[2, j],   fd_σσ[2]/H^2))
        push!(baroclinic_LHS, (row, imap[2, j+1], fd_σσ[3]/H^2))
        # term 2
        push!(baroclinic_LHS, (row, imap[1, j], -f/ν[j]))
    end

    # Upper boundary conditions: wind stress
    # b.c. 1: τξ = τξ_wind at σ = 0
    push!(baroclinic_LHS, (imap[1, nσ], imap[1, nσ], 1))
    # b.c. 2: τη = τη_wind at σ = 0
    push!(baroclinic_LHS, (imap[2, nσ], imap[2, nσ], 1))

    # Integral boundary conditions: transport
    # b.c. 1: -∫ σ τξ dσ = 1 or 0
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j],   -σ[j]  *(σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[1, 1], imap[1, j+1], -σ[j+2]*(σ[j+1] - σ[j])/2))
    end
    # b.c. 2: -∫ σ τη dσ = 1 or 0
    for j=1:nσ-1
        # trapezoidal rule
        push!(baroclinic_LHS, (imap[2, 1], imap[2, j],   -σ[j]  *(σ[j+1] - σ[j])/2))
        push!(baroclinic_LHS, (imap[2, 1], imap[2, j+1], -σ[j+2]*(σ[j+1] - σ[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    baroclinic_LHS = sparse((x->x[1]).(baroclinic_LHS), (x->x[2]).(baroclinic_LHS), (x->x[3]).(baroclinic_LHS), nvar*nσ, nvar*nσ)

    return lu(baroclinic_LHS)
end

function get_baroclinic_RHS(∂b∂x::Vector{Float64}, ∂b∂y::Vector{Float64}, τξ_wind::Real, τη_wind::Real, Uξ::Real, Uη::Real)
    nσ = size(σ, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    baroclinic_RHS = zeros(nvar*nσ)

    # eqtns:
    # eqtn 1: ∂σσ(τξ)/H² + f*τη/ν = 0 or ∂x(b)
    baroclinic_RHS[imap[1, 2:nσ-1]] = ∂b∂x[2:nσ-1] 
    # eqtn 2: ∂σσ(τη)/H² - f*τξ/ν = 0 or ∂y(b)
    baroclinic_RHS[imap[2, 2:nσ-1]] = ∂b∂y[2:nσ-1] 

    # top b.c.:
    # b.c. 1: τξ = τξ_wind at σ = 0
    baroclinic_RHS[imap[1, nσ]] = τξ_wind
    # b.c. 2: τη = τη_wind at σ = 0
    baroclinic_RHS[imap[2, nσ]] = τη_wind
    
    # integral b.c.:
    # b.c. 1: -∫ σ τξ dσ = 1 or 0
    baroclinic_RHS[imap[1, 1]] = Uξ
    # b.c. 2: -∫ σ τη dσ = 1 or 0
    baroclinic_RHS[imap[1, 1]] = Uη

    return baroclinic_RHS
end

function get_τξ_τη(baroclinic_LHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}, baroclinic_RHSs::Matrix{Float64})
    np = size(baroclinic_RHSs, 1)
    nσ = size(baroclinic_RHSs, 2)/2
    τ = zeros(np, 2*nσ)
    for i=1:np
        τ[i, :] = baroclinic_LHSs[i]\baroclinic_RHSs[i, :]
    end
    return τ[:, 1:nσ], τ[:, nσ+1:end]
end

function get_uξ_uη(τξ, τη, σ, H, ν)
    uξ = cumtrapz(H./ν.*τξ, σ)
    uη = cumtrapz(H./ν.*τη, σ)
    return uξ, uη
end
