####
# model problem: H⁻² ∂σσ(τ) - λ²τ = f
####

using nuPGCM
using SparseArrays
using LinearAlgebra
using PyPlot
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_m(p, t, C₀)
    # indices
	np = size(p, 1)
	nt = size(t, 1)

	# create global linear system using stamping method
    m = zeros(np)
	for k=1:nt
		# add contribution to m from element k
        for i=1:3
            func(ξ, η) = local_basis_func(C₀[k, :, i], [ξ, η])
            m[t[k, i]] += gaussian_quad2(func, p[t[k, :], :])
        end
	end

    return m
end

function get_M(p, t, C₀)
    np = size(p, 1)
    nt = size(t, 1)

	# create global linear system using stamping method
    M = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate M matrix element Mᵏ
        Mᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = local_basis_func(C₀[k, :, i], [ξ, η])*local_basis_func(C₀[k, :, j], [ξ, η])
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

function get_f(p, t, C₀, σ, f_func)
    np = size(p, 1)
    nt = size(t, 1)
    nσ = size(σ, 1)
    f = zeros(np, nσ)
    @showprogress "Calculating f..." for k=1:nt
        # get fᵏ
        fᵏ = zeros(3, nσ)
        for j=1:nσ
            g(ξ, η) = f_func(ξ, η, σ[j])
            for i=1:3
                func(ξ, η) = g(ξ, η)*local_basis_func(C₀[k, :, i], [ξ, η])
                fᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end
        # stamp
        f[t[k, :], :] .+= fᵏ
    end
    return f
end

function get_LHS(λ::Real, σ::AbstractArray{<:Real,1}, H::Real)
    nσ = size(σ, 1)
    LHS = Tuple{Int64,Int64,Float64}[]  

    # Interior nodes
    for j=2:nσ-1 
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # H⁻² ∂σσ(v) - λ²v = f
        # term 1
        push!(LHS, (j, j-1, fd_σσ[1]/H^2))
        push!(LHS, (j, j,   fd_σσ[2]/H^2))
        push!(LHS, (j, j+1, fd_σσ[3]/H^2))
        # term 2
        push!(LHS, (j, j, -λ^2))
    end

    # Upper boundary condition: v = v₀
    push!(LHS, (nσ, nσ, 1))
    # # Bottom boundary condition: v = 0
    # push!(LHS, (1, 1, 1))
    # integral boundary condition: -H² ∫ σ v dσ = Uφ
    for j=1:nσ-1
        # trapezoidal rule
        push!(LHS, (1, j,   -H^2 * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(LHS, (1, j+1, -H^2 * σ[j+1] * (σ[j+1] - σ[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), nσ, nσ)

    return lu(LHS)
end

function get_RHS(f::AbstractArray{<:Real,1}, v₀::Real, Uφ::Real)
    # ∂σσ(v) - λ²v = f
    RHS = copy(f)
    # Upper boundary condition: v = v₀
    RHS[end] = v₀
    # # Bottom boundary condition: v = 0
    # RHS[1] = 0
    # integral boundary condition: -H² ∫ σ v dσ = Uφ
    RHS[1] = Uφ
    return RHS
end

function test_problem()
    # mesh
    p, t, e = load_mesh("../meshes/circle2.h5")
    np = size(p, 1)
    Lx = 5e6
    Ly = 5e6
    p[:, 1] *= Lx
    p[:, 2] *= Ly
    ξ = p[:, 1]
    η = p[:, 2]
    C₀ = get_linear_basis_coeffs(p, t)

    # vertical coord
    nσ = 2^7
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  

    # decay
    λ = 1e-1

    # depth
    H₀ = 4e3
    # H = H₀*ones(np)
    Δ = Lx/5 
    G(x) = 1 - exp(-x^2/(2*Δ^2))
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    H = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + 100

    # wind stress
    τ₀ = zeros(np)

    # rhs function
    f_func(ξ, η, σ) = 0

    # LHS matrices
    LHSs = Array{Any}(undef, np)
    @showprogress "Calculating LHSs..." for i=1:np
        LHSs[i] = get_LHS(λ, σ, H[i])
    end

    # get m 
    m = get_m(p, t, C₀)

    # get M
    M = get_M(p, t, C₀)

    # get f
    # f = get_f(p, t, C₀, σ, f_func)
    f = zeros(np, nσ)

    # upper b.c.
    v₀ = M*τ₀

    # solve for v
    v = zeros(np, nσ)
    for i=1:np
        RHS = get_RHS(f[i, :], v₀[i], m[i])
        v[i, :] = LHSs[i]\RHS
    end
    println(m[990])
    println(-H[990]^2 * trapz(v[990, :] .* σ, σ))
    plot(v[990, :], σ); savefig("images/debug_v.png"); plt.close()

    # solve for τ
    τ = M\v
    plot(τ[990, :], σ); savefig("images/debug_t.png"); plt.close()

    ip = 100
    plot(τ[ip, :], σ)
    xlabel(L"$\tau$")
    ylabel(L"$\sigma$")
    savefig("images/tau_column$ip.png")
    println("images/tau_column$ip.png")
    plt.close()

    plot_horizontal(p, t, τ[:, 1]; clabel=L"$\tau(-1)$")
    savefig("images/tau_bot.png")
    println("images/tau_bot.png")
    plt.close()

    return τ
end

τ = test_problem()