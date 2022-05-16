####
# model problem: ∂σσ(τ) - λ²τ = f
####

using nuPGCM
using SparseArrays
using LinearAlgebra
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_M(p, t, e, C₀)
    np = size(p, 1)
    nt = size(t, 1)
    ne = size(e, 1)

	# create global linear system using stamping method
    M = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate M matrix element Mᵏ
        Mᵏ = get_Mᵏ(p, t, C₀, k)

		# add to global system
        for i=1:3
		    for j=1:3
                if t[k, i] in e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(M, (t[k, i], t[k, j], Mᵏ[i, j]))
			end
		end
	end

    # dirichlet
    for i=1:ne
        push!(M, (e[i], e[i], 1))
    end

    # make CSC matrix
    M = sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), np, np)

    return M
end

function get_Mᵏ(p, t, C₀, k)
    Mᵏ = zeros(3, 3)
    for i=1:3
        for j=1:3
            func(ξ, η) = local_basis_func(C₀[k, :, i], [ξ, η])*local_basis_func(C₀[k, :, j], [ξ, η])
            Mᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
        end
    end
    return Mᵏ
end

function get_fᵏ(p, t, C₀, k, f)
    fᵏ = zeros(3)
    for i=1:3
        func(ξ, η) = f(ξ, η)*local_basis_func(C₀[k, :, i], [ξ, η])
        fᵏ[i] = gaussian_quad2(func, p[t[k, :], :])
    end
    return fᵏ
end

function get_LHS(λ::Real, σ::AbstractArray{<:Real,1})
    nσ = size(σ, 1)
    LHS = Tuple{Int64,Int64,Float64}[]  

    # Interior nodes
    for j=2:nσ-1 
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # ∂σσ(u) - λ²u = f
        # term 1
        push!(LHS, (j, j-1, fd_σσ[1]))
        push!(LHS, (j, j,   fd_σσ[2]))
        push!(LHS, (j, j+1, fd_σσ[3]))
        # term 2
        push!(LHS, (j, j, -λ^2))
    end

    # Bottom boundary condition: u = 0
    push!(LHS, (1, 1, 1))
    # Upper boundary condition: u = u₀
    push!(LHS, (nσ, nσ, 1))

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), nσ, nσ)

    return lu(LHS)
end

function get_RHS(f::AbstractArray{<:Real,1}, u₀::Real)
    # ∂σσ(u) - λ²u = f
    RHS = copy(f)
    # Bottom boundary condition: u = 0
    RHS[1] = 0
    # Upper boundary condition: u = u₀
    RHS[end] = u₀
    return RHS
end

function test_problem()
    # mesh
    p, t, e = load_mesh("../meshes/circle1.h5")
    np = size(p, 1)
    nt = size(t, 1)
    ξ = p[:, 1]
    η = p[:, 2]
    C₀ = get_linear_basis_coeffs(p, t)

    # vertical coord
    nσ = 2^8
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  

    # parameters
    λ = 1e2
    τ₀(ξ, η) = 1 + 5*ξ - 2*η^2

    # rhs function
    f(ξ, η, σ) = 1 + 3*ξ + 5*η #+ 1e4*exp(-(σ + 0.1)^2)

    # LHS matrix
    LHS = get_LHS(λ, σ)

    # # solve for u
    # u = zeros(3, nt, nσ)
    # for k=1:nt
    #     # get fᵏ
    #     fᵏ = zeros(3, nσ)
    #     for j=1:nσ
    #         g(ξ, η) = f(ξ, η, σ[j])
    #         fᵏ[:, j] = get_fᵏ(p, t, C₀, k, g)
    #     end
    
    #     # get Mᵏ
    #     Mᵏ = get_Mᵏ(p, t, C₀, k)
    
    #     # upper b.c.
    #     u₀ = Mᵏ*τ₀.(ξ[t[k, :]], η[t[k, :]])
    
    #     # solve for u
    #     for i=1:3
    #         RHS = get_RHS(fᵏ[i, :], u₀[i])
    #         u[i, k, :] = LHS\RHS
    #     end
    # end
    
    # # get τ
    # τ = zeros(np, nσ)
    # for k=1:nt
    #     Mᵏ = get_Mᵏ(p, t, C₀, k)
    #     for j=1:nσ
    #         τ[t[k, :], j] .+= Mᵏ\u[:, k, j]
    #     end
    # end
    # for j=1:np
    #     # divide by number of triangles at τⱼ
    #     τ[j, :] /= count(x -> (x == j), t)
    # end

    # get M
    M = get_M(p, t, e, C₀)

    # upper b.c.
    u₀ = M*τ₀.(ξ, η)

    # solve for u
    u = zeros(np, nσ)
    for k=1:nt
        # get fᵏ
        fᵏ = zeros(3, nσ)
        for j=1:nσ
            g(ξ, η) = f(ξ, η, σ[j])
            fᵏ[:, j] = get_fᵏ(p, t, C₀, k, g)
        end

        # solve for u
        for i=1:3
            RHS = get_RHS(fᵏ[i, :], u₀[t[k, i]])
            u[t[k, i], :] .+= LHS\RHS
        end
    end

    # get τ
    τ = M\u

    # analytical solution
    τ_a(ξ, η, σ) = (τ₀(ξ, η) + f(ξ, η, σ)/λ^2)*exp(λ*σ) - f(ξ, η, σ)/λ^2

    # error
    τ_exact = zeros(np, nσ)
    for i=1:np
        τ_exact[i, :] = τ_a.(ξ[i], η[i], σ)
    end
    println("Max error: ", maximum(abs.(τ_exact .- τ)))

    ip = 100
    plot(τ[ip, :], σ, label="Numerical")
    plot(τ_exact[ip, :], σ, "--", label="Exact")
    legend()
    savefig("debug_vert.png")
    plt.close()

    plot_horizontal(p, t, abs.((τ[:, nσ] .- τ_exact[:, nσ])./τ_exact[:, nσ]); clabel=L"Relative Error at $\sigma = 0$")
    savefig("debug_horiz.png")
    plt.close()

    return τ
end

τ = test_problem()