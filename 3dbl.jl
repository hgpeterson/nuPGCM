using PyPlot, PyCall, Printf, LinearAlgebra

pl = pyimport("matplotlib.pylab")
lines = pyimport("matplotlib.lines")

# plotting stylesheet
plt.style.use("plots.mplstyle")
close("all")
pygui(false)

function plot_BL_sol()
    # parameters 
    f = -5.5e-5
    nσ = 2^11
    dσ = 1/(nσ - 3)
    σ = [-1-dσ; -1:dσ:0; dσ] # ghost nodes
    Hx = -2e-3
    Hy = -4e-3
    H = 2e3
    ν = 2e-3
    κ = 2e-3
    dbdξ_I = 2e-10
    dbdη_I = 1e-10
    uξ_I = -1e-3
    uη_I = -2e-3

    # analytical soluiton
    uξ_B_a, uη_B_a, b_B_a = analyticalBL(σ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)

    # numerical soluiton
    uξ_B_n, uη_B_n, b_B_n = numericalBL(σ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)

    # plot
    fig, ax = subplots(2, 3, figsize=(4, 4))

    ax[2, 1].set_xlabel(L"$u^\xi_\mathrm{B}$ ($\times 10^{-3}$ m s$^{-1}$)")
    ax[2, 2].set_xlabel(L"$u^\eta_\mathrm{B}$ ($\times 10^{-3}$ m s$^{-1}$)")
    ax[2, 3].set_xlabel(L"$b_\mathrm{B}$ ($\times 10^{-6}$ m s$^{-2}$)")

    ax[1, 1].set_ylabel(L"\sigma + 1")
    ax[2, 1].set_ylabel(L"\sigma + 1")
    ax[1, 1].set_ylim([0, 1])
    ax[1, 2].set_ylim([0, 1])
    ax[1, 3].set_ylim([0, 1])
    ax[1, 2].set_yticklabels([])
    ax[1, 3].set_yticklabels([])
    ax[2, 1].set_ylim([0, 0.02])
    ax[2, 2].set_ylim([0, 0.02])
    ax[2, 3].set_ylim([0, 0.02])
    ax[2, 1].set_yticks(0:0.01:0.02)
    ax[2, 2].set_yticks(0:0.01:0.02)
    ax[2, 3].set_yticks(0:0.01:0.02)
    ax[2, 2].set_yticklabels([])
    ax[2, 3].set_yticklabels([])
    ax[1, 1].spines["left"].set_visible(false)
    ax[1, 2].spines["left"].set_visible(false)
    ax[1, 3].spines["left"].set_visible(false)
    ax[2, 1].spines["left"].set_visible(false)
    ax[2, 2].spines["left"].set_visible(false)
    ax[2, 3].spines["left"].set_visible(false)

    ax[1, 1].axvline(0, lw=0.5, c="k", ls="-")
    ax[1, 1].plot(1e3*uξ_B_a, σ .+ 1, label="analytical")
    ax[1, 1].plot(1e3*uξ_B_n, σ .+ 1, ls="--", label="numerical")
    ax[1, 2].axvline(0, lw=0.5, c="k", ls="-")
    ax[1, 2].plot(1e3*uη_B_a, σ .+ 1, label="analytical")
    ax[1, 2].plot(1e3*uη_B_n, σ .+ 1, ls="--", label="numerical")
    ax[1, 3].axvline(0, lw=0.5, c="k", ls="-")
    ax[1, 3].plot(1e6*b_B_a, σ .+ 1, label="analytical")
    ax[1, 3].plot(1e6*b_B_n, σ .+ 1, ls="--", label="numerical")

    ax[2, 1].axvline(0, lw=0.5, c="k", ls="-")
    ax[2, 1].plot(1e3*uξ_B_a, σ .+ 1, label="analytical")
    ax[2, 1].plot(1e3*uξ_B_n, σ .+ 1, ls="--", label="numerical")
    ax[2, 2].axvline(0, lw=0.5, c="k", ls="-")
    ax[2, 2].plot(1e3*uη_B_a, σ .+ 1, label="analytical")
    ax[2, 2].plot(1e3*uη_B_n, σ .+ 1, ls="--", label="numerical")
    ax[2, 3].axvline(0, lw=0.5, c="k", ls="-")
    ax[2, 3].plot(1e6*b_B_a, σ .+ 1, label="analytical")
    ax[2, 3].plot(1e6*b_B_n, σ .+ 1, ls="--", label="numerical")

    ax[1, 1].legend()

    savefig("BL_sol.png")
    println("BL_sol.png")
    plt.close()
end

function get_params(f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I)
    S = -(Hx*dbdξ_I + Hy*dbdη_I)/f^2
    T = (Hy*dbdξ_I - Hx*dbdη_I)/f^2
    ε = sqrt(ν/abs(f)/H^2)
    δ = sqrt(2*ν/abs(f))
    μ = ν/κ
    return S, T, ε, δ, μ
end

function numericalBL(σ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)
    # params
    S, T, ε, δ, μ = get_params(f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I)
    dσ = σ[2] - σ[1]
    nσ = size(σ, 1)
    nVars = 3 # uξ, uη, b
    nPts = nσ*nVars

    # map
    imap = reshape(1:nPts, (nVars, nσ))

    # Ax = y
    A = zeros(nPts, nPts)
    y = zeros(nPts)
    for i=2:nσ-1
        # -uη + 1/f ∂ₓH b - ε^2 ∂σσ(uξ) = 0
        A[imap[1, i], imap[1, i-1]] =  -ε^2/dσ^2
        A[imap[1, i], imap[1, i]]   = 2*ε^2/dσ^2
        A[imap[1, i], imap[1, i+1]] =  -ε^2/dσ^2
        A[imap[1, i], imap[2, i]] = -1
        A[imap[1, i], imap[3, i]] = Hx/f

        # uξ - ε^2 ∂σσ(uη) = 0
        A[imap[2, i], imap[1, i]] = 1
        A[imap[2, i], imap[2, i-1]] =  -ε^2/dσ^2
        A[imap[2, i], imap[2, i]]   = 2*ε^2/dσ^2
        A[imap[2, i], imap[2, i+1]] =  -ε^2/dσ^2

        # uξ dξ(bI) + uη dη(bI) - κ/H^2 ∂σσ(b) = 0
        A[imap[3, i], imap[1, i]] = dbdξ_I
        A[imap[3, i], imap[2, i]] = dbdη_I
        A[imap[3, i], imap[3, i-1]] =  -κ/H^2/dσ^2
        A[imap[3, i], imap[3, i]]   = 2*κ/H^2/dσ^2
        A[imap[3, i], imap[3, i+1]] =  -κ/H^2/dσ^2
    end
    # uξ = -uξ_I at σ = -1
    A[imap[1, 1], imap[1, 2]] = 1
    y[imap[1, 1]] = -uξ_I
    # uη = -uη_I at σ = -1
    A[imap[2, 1], imap[2, 2]] = 1
    y[imap[2, 1]] = -uη_I
    # # Uξ dbdξ_I + Uη dbdη_I + κ/H ∂σ(b) = 0 at σ = -1
    # A[imap[3, 1], imap[3, 1]] = -κ/H * 1/(2*dσ)
    # A[imap[3, 1], imap[3, 3]] =  κ/H * 1/(2*dσ)
    # for i=2:nσ-2
    #     A[imap[3, 1], imap[1, i]] += 1/2*dbdξ_I*dσ
    #     A[imap[3, 1], imap[1, i+1]] += 1/2*dbdξ_I*dσ
    #     A[imap[3, 1], imap[2, i]] += 1/2*dbdη_I*dσ
    #     A[imap[3, 1], imap[2, i+1]] += 1/2*dbdη_I*dσ
    # end
    # ∂σ(b) = 0 at σ = 0
    A[imap[3, 1], imap[3, end]] =  1/(2*dσ)
    A[imap[3, 1], imap[3, end-2]] = -1/(2*dσ)

    # uξ = 0 at σ = 0
    A[imap[1, end], imap[1, end-1]] = 1
    # uη = 0 at σ = 0
    A[imap[2, end], imap[2, end-1]] = 1
    # b = 0 at σ = 0
    A[imap[3, end], imap[3, end-1]] = 1

    # println(rank(A))
    # println(nPts)

    # solve
    sol = A\y

    # unpack
    uξ = sol[imap[1, :]]
    uη = sol[imap[2, :]]
    b = sol[imap[3, :]]

    return uξ, uη, b
end

function analyticalBL(σ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)
    S, T, ε, δ, μ = get_params(f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I)

    p, q = get_pq(S, T, δ, μ)

    c1 = -uξ_I
    c2 = (uξ_I*(q^2 - p^2) - uη_I*δ^2*(p^2 + q^2)^2/2)/(2*p*q)
    c3 = ( 2*c2*p*q + c1*(q^2 - p^2))/(δ^2*(p^2 + q^2)^2/2)
    c4 = (-2*c1*p*q + c2*(q^2 - p^2))/(δ^2*(p^2 + q^2)^2/2)
    c5 = f/Hx * (c3 + δ^2*(c1*(q^2 - p^2) - 2*c2*p*q)/2)
    c6 = f/Hx * (c4 + δ^2*(c2*(q^2 - p^2) + 2*c1*p*q)/2)

    uξ_B = @. exp(-q*H*(σ + 1))*(c1*cos(p*H*(σ + 1)) + c2*sin(p*H*(σ + 1)))
    uη_B = @. exp(-q*H*(σ + 1))*(c3*cos(p*H*(σ + 1)) + c4*sin(p*H*(σ + 1)))
    b_B  = @. exp(-q*H*(σ + 1))*(c5*cos(p*H*(σ + 1)) + c6*sin(p*H*(σ + 1)))
    # b_B = zeros(size(uξ_B))
    # dσ = σ[2] - σ[1]
    # b_B[2:end-1] = @. f/Hx*(uη_B[2:end-1] + ε^2*(uξ_B[3:end] - 2*uξ_B[2:end-1] + uξ_B[1:end-2])/dσ^2)
    return uξ_B, uη_B, b_B
end

function get_pq(S, T, δ, μ)
    r = (-1 + im*sqrt(3))/2
    # r = (-1 - im*sqrt(3))/2
    c = sqrt(μ^2*T^2/4 + (1 + μ*S)^3/27)
    λ = sqrt(2)/δ * sqrt(r*cbrt(-μ*T/2 + c) + conj(r)*cbrt(-μ*T/2 - c))
    q = real(λ)
    p = imag(λ)
    # c = sqrt(μ^2*T^2/4 + (1 + μ*S)^3/27)
    # for r=[1, (-1 + im*sqrt(3))/2, (-1 - im*sqrt(3))/2]
    #     λ2 = 2/δ^2 * (r*cbrt(-μ*T/2 + c) + conj(r)*cbrt(-μ*T/2 - c))
    #     # println(real(λ2))
    #     # println(imag(λ2))

    #     λ = sqrt(2)/δ * sqrt(r*cbrt(-μ*T/2 + c) + conj(r)*cbrt(-μ*T/2 - c))
    #     q = -real(λ)
    #     p = imag(λ)
    #     println(q)
    #     println(p)
    #     println()
    # end
    return p, q
end

plot_BL_sol()

# get_pq(0.001, 0.001, 10, 1)
