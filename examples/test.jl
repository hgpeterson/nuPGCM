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

function get_Mᵏ(p, t, C₀, k)
    Mᵏ = 0
    for i=1:3
        func(ξ, η) = local_basis_func(C₀[k, :, i], [ξ, η])*local_basis_func(C₀[k, :, 1], [ξ, η])
        Mᵏ += gaussian_quad2(func, p[t[k, :], :])
    end
    return Mᵏ
end

function get_fᵢ(p, t, C₀, k, f)
    fᵢ = 0
    for i=1:3
        func(ξ, η) = f(ξ, η)*local_basis_func(C₀[k, :, i], [ξ, η])
        fᵢ += gaussian_quad2(func, p[t[k, :], :])
    end
    return fᵢ
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

function get_u(LHSs::AbstractArray{Any,1}, RHSs::AbstractArray{<:Real,2})
    nt = size(RHSs, 1)
    nσ = size(RHSs, 2)
    u = zeros(nt, nσ)
    for k=1:nt
        u[k, :] = LHSs[k]\RHSs[k, :]
    end
    return u
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
    λ = 1e1
    τ₀ = 1

    # rhs function
    f(ξ, η, σ) = 1

    # LHS matrices
    LHSs = Array{Any}(undef, nt) 
    for k=1:nt 
        LHSs[k] = get_LHS(λ, σ)
    end  

    # RHS vectors
    RHSs = zeros(nt, nσ)
    for k=1:nt
        fᵢ = zeros(nσ)
        for j=1:nσ
            g(ξ, η) = f(ξ, η, σ[j])
            fᵢ[j] = get_fᵢ(p, t, C₀, k, g)
        end
        Mᵏ = get_Mᵏ(p, t, C₀, k)
        u₀ = Mᵏ*τ₀ #FIXME: when you sum over j, this gets double counted
        RHSs[k, :] = get_RHS(fᵢ, u₀)
    end

    # solve for u = Mᵏτ
    u = get_u(LHSs, RHSs)

    # get τ
    τ = zeros(np, nσ)
    for k=1:nt
        Mᵏ = get_Mᵏ(p, t, C₀, k)
        for j=1:3
            τ[t[k, j], :] .+= u[k, :]/Mᵏ
        end
    end

    # analytical solution
    τ_exact(ξ, η, σ) = (τ₀ + f(ξ, η, σ)/λ^2)*exp(λ*σ) - f(ξ, η, σ)/λ^2

    # error
    ip = 100
    τ_exact_ip = τ_exact.(ξ[ip], η[ip], σ)
    println(maximum(abs.(τ_exact_ip .- τ[ip, :])))

    plot(τ[ip, :], σ, label="Numerical")
    plot(τ_exact_ip, σ, label="Exact")
    savefig("debug.png")
    plt.close()

    return τ
end

test_problem()