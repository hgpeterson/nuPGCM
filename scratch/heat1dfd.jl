using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function get_K(σ)
    nσ = length(σ)
    K = Tuple{Int64,Int64,Float64}[]
    for j=2:nσ-1
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        push!(K, (j, j-1, fd_σσ[1]))
        push!(K, (j, j  , fd_σσ[2]))
        push!(K, (j, j+1, fd_σσ[3]))
    end
    fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
    push!(K, (1, 1, fd_σ[1]))
    push!(K, (1, 2, fd_σ[2]))
    push!(K, (1, 3, fd_σ[3]))
    fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    push!(K, (nσ, nσ-2, fd_σ[1]))
    push!(K, (nσ, nσ-1, fd_σ[2]))
    push!(K, (nσ, nσ  , fd_σ[3]))
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nσ, nσ))
end

function solve_heat()
    nσ = 2^7
    σ = -1:1/(nσ-1):0 
    # σ = -(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2 

    ε² = 1e-2
    μ = 1
    ϱ = 1e-4
    H = 1
    α = ε²/H^2/μ/ϱ
    T = 0.2/α
    n_steps = 2^2
    Δt = T/n_steps

    # b = H*σ
    b = @. cos(π*σ)

    K = get_K(σ)
    LHS = I - α*Δt/2*K
    LHS[1, :] = K[1, :]
    LHS[nσ, :] = K[nσ, :]
    LHS = lu(LHS)
    RHS = I + α*Δt/2*K

    # f(σ, t) = 0
    f(σ, t) = α*exp(-α*t)*(π^2 - H^2)*cos(π*σ)/H^2
    for i=1:n_steps
        r = RHS*b + Δt/2*(f.(σ, (i-1)*Δt) + f.(σ, i*Δt))
        r[1] = 0
        r[nσ] = 0
        b = LHS\r
    end

    ba = @. cos(π*σ)*exp(-α*T)
    err_abs = abs.(b - ba)
    err_rel = abs.(b - ba)./ba
    println(@sprintf("Max Abs Error: %1.1e at i=%d", maximum(err_abs), argmax(err_abs)))
    println(@sprintf("Max Rel Error: %1.1e at i=%d", maximum(err_rel), argmax(err_rel)))

    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
    ax[1].plot(b, σ, label="Numerical")
    ax[1].plot(ba, σ, "--", label="Analytical")
    ax[1].set_xlim(-1, 1)
    ax[1].set_xlabel(L"b")
    ax[1].set_ylabel(L"\sigma")
    ax[2].semilogx(abs.(b - ba), σ)
    ax[2].set_xlabel("Absolute error")
    ax[1].legend()
    savefig("scratch/images/b.png")
    println("scratch/images/b.png")
    plt.close()
end

solve_heat()