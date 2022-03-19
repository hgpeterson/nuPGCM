function get_barotropic_LHS(p, t, e, f₀, β, H, Hx, Hy, τₜ_ξ, τₜ_η)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    C = Tuple{Int64,Int64,Float64}[]  
    K = Tuple{Int64,Int64,Float64}[]  
	for k=1:n_tri
		# get coeffs for linear basis f c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
		V = zeros(3, 3)
		for row=1:3
			V[row, :] = [1 p[t[k, row], 1] p[t[k, row], 2]]
		end
		C₀ = inv(V)

		# calculate C matrix elements Cₑ
        Cₑ = zeros(3, 3)
        for i=1:3
            for j=1:3
                f(p₀) = -(f₀ + β*p₀[2])*Hx(p₀[1], p₀[2])/H(p₀[1], p₀[2])^2*C₀[3, j]*local_basis_func(C₀[:, i]', p₀) -
                    (β/H(p₀[1], p₀[2]) - (f₀ + β*p₀[2])*Hy(p₀[1], p₀[2])/H(p₀[1], p₀[2])^2)*C₀[2, j]*local_basis_func(C₀[:, i]', p₀)
                Cₑ[i, j] = gaussian_quad2(f, p[t[k, :], :])
            end
        end

		# calculate K matrix element Kₑ
        Kₑ = zeros(3, 3)
        for i=1:3
            for j=1:3
                f(p₀) = τₜ_η(p₀[1], p₀[2])/H(p₀[1], p₀[2])*C₀[2, j]*C₀[2, i] + 
                           τₜ_ξ(p₀[1], p₀[2])/H(p₀[1], p₀[2])*C₀[3, j]*C₀[3, i]
                Kₑ[i, j] = gaussian_quad2(f, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                push!(C, (imap[t[k, i]], imap[t[k, j]], Cₑ[i, j]))
                push!(K, (imap[t[k, i]], imap[t[k, j]], Kₑ[i, j]))
			end
		end
	end
    
    # make CSC matrices
    C = sparse((x->x[1]).(C), (x->x[2]).(C), (x->x[3]).(C), n_nodes, n_nodes)
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), n_nodes, n_nodes)

    # full barotropic_LHS matrix
    barotropic_LHS = C + K
  
    # dirichlet Ψ = 0 along edges
    for i=1:size(e, 1)
        row = imap[e[i]]
        barotropic_LHS[row, :] .= 0
        barotropic_LHS[row, row] = 1
    end
	dropzeros!(barotropic_LHS)

    return barotropic_LHS
end


function get_barotropic_RHS(p, t, e, F)
	n_nodes = size(p, 1)
	n_tri = size(t, 1)
    imap = reshape(1:n_nodes, 1, n_nodes) 

	# create global linear system using stamping method
    barotropic_RHS = zeros(n_nodes)
	for k=1:n_tri
		# get coeffs for linear basis f c₁ + c₂ξ + c₃η stored in 3×3 C₀ matrix
		V = zeros(3, 3)
		for row = 1:3
			V[row, :] = [1 p[t[k, row], 1] p[t[k, row], 2]]
		end
		C₀ = inv(V)

		# calculate barotropic_RHS vector element and add it to the global system
        for i=1:3
            f(p₀) = F(p₀[1], p₀[2])*local_basis_func(C₀[:, i]', p₀)
            barotropic_RHS[imap[t[k, i]]] += gaussian_quad2(f, p[t[k, :], :])
        end
	end
    
    # dirichlet Ψ = 0 along edges
    for i=1:size(e, 1)
        row = imap[e[i]]
        barotropic_RHS[row] = 0
    end

    return barotropic_RHS
end