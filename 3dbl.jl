using PyPlot, PyCall, Printf, LinearAlgebra

pl = pyimport("matplotlib.pylab")
lines = pyimport("matplotlib.lines")

# plotting stylesheet
plt.style.use("plots.mplstyle")
close("all")
pygui(false)

function plot_uξ()
    # parameters 
    f = -5.5e-5
    # dζ = 0.001
    # H = 1
    dζ = 0.01
    H = 0.1
    ζ = @. 0:dζ:H
    Hx = -2e-3
    Hy = -4e-3
    H = 4e3
    ν = 2e-2
    κ = 2e-3
    dbdξ_I = 2e-10
    dbdη_I = 1e-10
    uξ_I = 1e-3
    uη_I = 2e-3

    # analytical soluiton
    uξ_B_a = analyticalBL(ζ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)

    # numerical soluiton
    uξ_B_n = numericalBL(ζ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)

    # plot
    fig, ax = subplots(1)
    ax.set_xlabel(L"BL $u^\xi$ (mm s$^{-1}$)")
    ax.set_ylabel(L"\zeta")
    ax.axvline(0, lw=0.5, c="k", ls="-")
    ax.plot(1e3*uξ_B_a, ζ, label="analytical")
    ax.plot(1e3*uξ_B_n, ζ, ls="--", label="numerical")
    ax.set_ylim([0, 0.1])
    ax.legend()
    savefig("uB.png", bbox_inches="tight")
    println("uB.png")
end

function get_params(f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I)
    S = -(Hx*dbdξ_I + Hy*dbdη_I)/f^2
    T = (Hy*dbdξ_I - Hx*dbdη_I)/f^2
    ε = sqrt(ν/abs(f)/H^2)
    μ = ν/κ
    return S, T, ε, μ
end

function numericalBL(ζ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)
    # params
    S, T, ε, μ = get_params(f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I)
    F = μ*Hx^2/f^2 * (uξ_I*dbdξ_I + uη_I*dbdη_I) + uξ_I*Hx + uη_I*Hy
    F /= Hx*ε^4
    dζ = ζ[2] - ζ[1]
    n = size(ζ, 1)

    # matrix A, vector b
    A = zeros(n, n)
    b = zeros(n)
    for i=4:n-3
        # u^(6)
        A[i, i-3] =   1/dζ^6
        A[i, i-2] =  -6/dζ^6
        A[i, i-1] =  15/dζ^6
        A[i, i]   = -20/dζ^6
        A[i, i+1] =  15/dζ^6
        A[i, i+2] =  -6/dζ^6
        A[i, i+3] =   1/dζ^6

        # (1 + μS)/ε^4 u''
        A[i, i-1] +=  (1 + μ*S)/ε^4 *  1/dζ^2
        A[i, i]   +=  (1 + μ*S)/ε^4 * -2/dζ^2
        A[i, i+1] +=  (1 + μ*S)/ε^4 *  1/dζ^2

        # μT/ε^6 u
        A[i, i] += μ*T/ε^6
    end
    # u = -uI at ζ = 0
    A[1, 1] = 1
    b[1] = -uξ_I
    # u^(4) + Hy/Hx/ε^2 u^(2) = F at ζ = 0
    A[2, 1] =   3/dζ^4
    A[2, 2] = -14/dζ^4
    A[2, 3] =  26/dζ^4
    A[2, 4] = -24/dζ^4
    A[2, 5] =  11/dζ^4
    A[2, 6] =  -2/dζ^4
    A[2, 1] += Hy/Hx/ε^2 *  2/dζ^2
    A[2, 2] += Hy/Hx/ε^2 * -5/dζ^2
    A[2, 3] += Hy/Hx/ε^2 *  4/dζ^2
    A[2, 4] += Hy/Hx/ε^2 * -1/dζ^2
    b[2] = F
    # u''' = 0 at ζ = H 
    A[3, n]   =  5/2 /dζ^3
    A[3, n-1] =    9 /dζ^3
    A[3, n-2] =  -12 /dζ^3
    A[3, n-3] =    7 /dζ^3
    A[3, n-4] = -3/2 /dζ^3

    # u = 0 at ζ = H
    A[n, n] = 1
    # u' = 0 at ζ = H
    A[n-1, n]   = 3/2 /dζ
    A[n-1, n-1] =  -2 /dζ
    A[n-1, n-2] = 1/2 /dζ
    # u'' = 0 at ζ = H
    A[n-2, n]   =  2/dζ^2
    A[n-2, n-1] = -5/dζ^2
    A[n-2, n-2] =  4/dζ^2
    A[n-2, n-3] = -1/dζ^2

    # println(rank(A))
    # println(n)

    # imshow(log.(abs.(A)))
    # colorbar()
    # savefig("A.png")
    # plt.close()

    # display(nullspace(A))

    # solve
    uξ_B = A\b

    return uξ_B
end

function analyticalBL(ζ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)
    S, T, ε, μ = get_params(f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I)

    p, q = get_pq(S, T, ε, μ)

    c1 = -uξ_I
    F = μ*Hx^2/f^2 * (uξ_I*dbdξ_I + uη_I*dbdη_I) + uξ_I*Hx + uη_I*Hy
    c2 = 2*ε^2*q^2*Hx/Hy * uξ_I + F/(2*Hy*ε^2*q^2)

    uξ_B = @. exp(-q*ζ)*(c1*cos(p*ζ) + c2*sin(p*ζ))
    return uξ_B
end

function get_pq(S, T, ε, μ)
    r = (-1 + im*sqrt(3))/2
    c = sqrt(μ^2*T^2/4 + (1 + μ*S)^3/27)
    λ = 1/ε * sqrt(r*cbrt(-μ*T/2 + c) + conj(r)*cbrt(-μ*T/2 - c))
    q = real(λ)
    p = imag(λ)
    return p, q
end

function plot_pq()
    ε = 1e-3
    μ = 1
    Ss = 10. .^(-5:0.1:2)
    Ts = -10. .^(-4:2)
    cs = pl.cm.viridis(range(1, 0, length=size(Ts, 1)))
    ls = ["-", "--", ":", "-.", ":", "--", "-"]

    fig, ax = subplots(1, 2, figsize=(6.5, 6.5/1.62/2), sharey=true)
    for i=1:size(Ts, 1)
        T = Ts[i]
        c = cs[i, :] 
        label = string(L"$T = $", @sprintf("%1.0e", T))
        pq = get_pq.(Ss, T, ε, μ)
        ax[1].semilogx(Ss, ε*last.(pq),  c=c, ls=ls[i], label=label)
        ax[2].semilogx(Ss, ε*first.(pq), c=c, ls=ls[i], label=label)
    end
    ax[2].legend(loc=(1.1, 0.1))
    ax[1].set_xlabel(L"$S$")
    ax[2].set_xlabel(L"$S$")
    ax[1].set_ylabel(L"$\varepsilon q$")
    ax[2].set_ylabel(L"$\varepsilon p$")
    ax[1].set_ylim([0.5, 2.5])
    tight_layout()
    savefig("pq.png")
    println("pq.png")
end

plot_uξ()
# plot_pq()