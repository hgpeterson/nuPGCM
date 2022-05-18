####
# model problem: ∂σσ(τ) - λ²τ = f
####

using nuPGCM
using SparseArrays
using LinearAlgebra
using PyPlot
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_M(p, t, C₀)
    np = size(p, 1)
    nt = size(t, 1)

	# create global linear system using stamping method
    M = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate M matrix element Mᵏ
        Mᵏ = get_Mᵏ(p, t, C₀, k)

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

function get_f(p, t, C₀, σ, f_func)
    np = size(p, 1)
    nt = size(t, 1)
    nσ = size(σ, 1)
    f = zeros(np, nσ)
    # @@@@ this is the slow part because it needs to integrate over every triangle and every vertical level
    @showprogress "Calculating f..." for k=1:nt
        # get fᵏ
        fᵏ = zeros(3, nσ)
        for j=1:nσ
            g(ξ, η) = f_func(ξ, η, σ[j])
            fᵏ[:, j] = get_fᵏ(p, t, C₀, k, g)
        end
        # stamp
        f[t[k, :], :] .+= fᵏ
    end
    return f
end

function get_fᵏ(p, t, C₀, k, f_func)
    fᵏ = zeros(3)
    for i=1:3
        func(ξ, η) = f_func(ξ, η)*local_basis_func(C₀[k, :, i], [ξ, η])
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

        # ∂σσ(v) - λ²v = f
        # term 1
        push!(LHS, (j, j-1, fd_σσ[1]))
        push!(LHS, (j, j,   fd_σσ[2]))
        push!(LHS, (j, j+1, fd_σσ[3]))
        # term 2
        push!(LHS, (j, j, -λ^2))
    end

    # Bottom boundary condition: v = 0
    push!(LHS, (1, 1, 1))
    # Upper boundary condition: v = v₀
    push!(LHS, (nσ, nσ, 1))

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), nσ, nσ)

    return lu(LHS)
end

function get_RHS(f::AbstractArray{<:Real,1}, v₀::Real)
    # ∂σσ(v) - λ²v = f
    RHS = copy(f)
    # Bottom boundary condition: v = 0
    RHS[1] = 0
    # Upper boundary condition: v = v₀
    RHS[end] = v₀
    return RHS
end

function test_problem()
    # mesh
    p, t, e = load_mesh("../meshes/circle2.h5")
    np = size(p, 1)
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
    f_func(ξ, η, σ) = 1 + 3*ξ + 5*η #+ 1e4*exp(-(σ + 0.1)^2)

    # LHS matrix
    LHS = get_LHS(λ, σ)

    # get M
    M = get_M(p, t, C₀)

    # get f
    f = get_f(p, t, C₀, σ, f_func)

    # upper b.c.
    v₀ = M*τ₀.(ξ, η)

    # solve for v
    v = zeros(np, nσ)
    for i=1:np
        RHS = get_RHS(f[i, :], v₀[i])
        v[i, :] = LHS\RHS
    end

    # solve for τ
    τ = M\v

    # analytical solution
    τ_a(ξ, η, σ) = (τ₀(ξ, η) + f_func(ξ, η, σ)/λ^2)*exp(λ*σ) - f_func(ξ, η, σ)/λ^2

    # error
    τ_exact = zeros(np, nσ)
    for i=1:np
        τ_exact[i, :] = τ_a.(ξ[i], η[i], σ)
    end
    println("Max abs error: ", maximum(abs.(τ_exact .- τ)))

    ip = 100
    plot(τ[ip, :], σ, label="Numerical")
    plot(τ_exact[ip, :], σ, "--", label="Exact")
    legend()
    savefig("tau$ip.png")
    plt.close()

    jlevs = [nσ, nσ - 32, nσ - 64]
    # jlevs = [nσ]
    for j=jlevs
        plot_horizontal(p, t, abs.((τ[:, j] .- τ_exact[:, j])); clabel=latexstring(L"Absolute Error at $\sigma =$", σ[j]))
        savefig("ae$j.png")
        plt.close()
        plot_horizontal(p, t, f_func.(ξ, η, σ[j]); clabel=latexstring(L"$f$ at $\sigma =$", σ[j]))
        savefig("f$j.png")
        plt.close()
    end
    plot_horizontal(p, t, τ₀.(ξ, η); clabel=L"$\tau_0$")
    savefig("tau0.png")
    plt.close()

    return τ
end

τ = test_problem()