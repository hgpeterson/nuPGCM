using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function b_a(σ, t, α, H; N=1000)
    # H*σ, nuemann
    A(n) = 2*H*(1 + (-1)^(n+1))/(n^2*π^2)
    return -H/2 + sum(A(n)*cos(n*π*σ)*exp(-α*n^2*π^2*t) for n=1:2:N)

    # # H^3*(σ^2 + 2/3*σ^3), nuemann
    # A(n) = 8*H^3*(-1 + (-1)^n)/(n^4*π^4)
    # return H^3/6 + sum(A(n)*cos(n*π*σ)*exp(-α*n^2*π^2*t) for n=1:2:N)
end

function solve_heat()
    nσ = 2^8
    σ = -1:1/(nσ-1):0 
    # σ = -(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2 
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g = Grid(1, σ, t, e)

    # ε² = 1e-2
    # μ = 1
    # ϱ = 1e-4
    # H = 1
    # α = ε²/H^2/μ/ϱ
    H = 1e3
    α = 1e-4/H^2
    println(α)
    T = 1e-3/α
    println(T/86400)
    n_steps = 2^5
    Δt = T/n_steps
    println(Δt/86400)

    # δ = 2e-2
    # b = @. H*σ + δ*exp(-H*(σ + 1)/δ) - δ*exp(H*σ/δ)
    # b = H*σ
    # b = @. H^3*(σ^2 + 2/3*σ^3)
    # b = @. cos(π*σ)
    b = 1e-6*H*σ

    M = nuPGCM.mass_matrix(g)
    K = nuPGCM.stiffness_matrix_zz(g)
    LHS = M + α*Δt/2*K
    LHS = lu(LHS)
    RHS = M - α*Δt/2*K

    f(σ, t) = 0
    # f(σ, t) = α*exp(-α*t)*(π^2 - H^2)*cos(π*σ)/H^2
    for i=1:n_steps
        r = RHS*b + M*(Δt/2*(f.(σ, (i-1)*Δt) + f.(σ, i*Δt)))
        b = LHS\r
    end

    # ba = [b_a(σ[i], T, α, H) for i ∈ eachindex(σ)]
    ba = [1e-6*b_a(σ[i], T, α, H) for i ∈ eachindex(σ)]
    # ba = @. cos(π*σ)*exp(-α*T)
    err_abs = FEField(abs.(b - ba), g)
    err_rel = FEField(abs.(b - ba)./ba, g)
    println(@sprintf("Max Abs Error: %1.1e at i=%d", maximum(err_abs), argmax(err_abs)))
    println(@sprintf("Max Rel Error: %1.1e at i=%d", maximum(err_rel), argmax(err_rel)))
    println(@sprintf("L2 Error: %1.1e", L2norm(err_abs)))

    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
    ax[1].plot(b, σ, label="Numerical")
    ax[1].plot(ba, σ, "--", label="Analytical")
    # ax[1].set_xlim(-1, 0)
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