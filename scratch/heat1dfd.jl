using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function get_K(σ, κ)
    nσ = length(σ)
    K = Tuple{Int64,Int64,Float64}[]
    for j=2:nσ-1
        fd_σ  = mkfdstencil(σ[j-1:j+1], σ[j], 1)
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        κ_σ = fd_σ'*κ[j-1:j+1]
        # ∂σ(κ ∂σ(b)) = κ ∂σσ(b) + ∂σ(κ) ∂σ(b)
        push!(K, (j, j-1, κ[j]*fd_σσ[1] + κ_σ*fd_σ[1]))
        push!(K, (j, j,   κ[j]*fd_σσ[2] + κ_σ*fd_σ[2]))
        push!(K, (j, j+1, κ[j]*fd_σσ[3] + κ_σ*fd_σ[3]))
    end
    # ∂σ(b) = 0 at z = -H
    fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
    push!(K, (1, 1, fd_σ[1]))
    push!(K, (1, 2, fd_σ[2]))
    push!(K, (1, 3, fd_σ[3]))
    # ∂σ(b) = 0 at z = 0
    fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    push!(K, (nσ, nσ-2, fd_σ[1]))
    push!(K, (nσ, nσ-1, fd_σ[2]))
    push!(K, (nσ, nσ  , fd_σ[3]))
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nσ, nσ))
end

function solve_heat()
    # params
    nσ = 2^4
    σ = -1:1/(nσ-1):0 
    # σ = -(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2 
    ε² = 1e-2
    μ = 1
    ϱ = 1e-4
    H = 1.0
    κ = @. 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ = ones(nσ)
    # α = ε²/μ/ϱ
    α = 1e-2
    # T = 5e-2/α
    # n_steps = 50
    # Δt = T/n_steps
    Δt = 0.04
    n_steps = 1000

    # b₀
    b = H*σ
    ∫b₀ = trapz(b, σ*H)
    println("∫b₀ = ", ∫b₀)

    # LHS and RHS
    K = get_K(σ, κ)
    LHS = I - α/H^2*Δt/2*K
    LHS[1, :] = K[1, :]
    LHS[nσ, :] = K[nσ, :]
    LHS = lu(LHS)
    RHS = I + α/H^2*Δt/2*K

    # loop
    for i=1:n_steps
        r = RHS*b
        r[1] = 0
        r[nσ] = 0
        b = LHS\r
    end

    # integral conservation
    ∫b = trapz(b, σ*H)
    println("∫b = ", ∫b)
    Δb = abs(∫b - ∫b₀)
    println("Δb = ", Δb)
    Δb_pct = 100*Δb/abs(∫b₀)
    println("Δb_pct = ", Δb_pct)

    # plot
    bz = differentiate(b, σ*H)
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
    ax[1].plot(b, σ*H)
    ax[1].set_xlim(-H, 0)
    ax[2].plot(bz, σ*H)
    ax[1].set_xlabel(L"b")
    ax[2].set_xlabel(L"\partial_z b")
    ax[1].set_ylabel(L"z")
    savefig("images/b.png")
    println("images/b.png")
    plt.close()
end

solve_heat()