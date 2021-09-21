using PyPlot, PyCall, Printf

pl = pyimport("matplotlib.pylab")
lines = pyimport("matplotlib.lines")

# plotting stylesheet
plt.style.use("plots.mplstyle")
close("all")
pygui(false)

function testrun()
    # parameters 
    f = -5.5e-5
    nσ = 2^8
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
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
    uξ_B = analyticalBL(σ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)

    # plot
    fig, ax = subplots(1)
    ax.set_xlabel(L"BL $u^\xi$ (mm s$^{-1}$)")
    ax.set_ylabel(L"\sigma")
    ax.axvline(0, lw=0.5, c="k", ls="-")
    ax.set_ylim([-1, -0.9])
    ax.plot(1e3*uξ_B, σ)
    tight_layout()
    savefig("uB.png")
    println("uB.png")
end

function analyticalBL(σ, f, Hx, Hy, H, ν, κ, dbdξ_I, dbdη_I, uξ_I, uη_I)
    S = -(Hx*dbdξ_I + Hy*dbdη_I)/f^2
    T = (Hy*dbdξ_I - Hx*dbdη_I)/f^2
    ϵ = sqrt(ν/abs(f)/H^2)
    μ = ν/κ

    p, q = get_pq(S, T, ϵ, μ)

    c1 = -uξ_I
    F = μ*Hx^2/f^2 * (uξ_I*dbdξ_I + uη_I*dbdη_I) + uξ_I*Hx + uη_I*Hy
    c2 = 2*ϵ^2*q^2*Hx/Hy * uξ_I + F/(2*Hy*ϵ^2*q^2)

    ζ = σ .+ 1
    uξ_B = @. exp(-q*ζ)*(c1*cos(p*ζ) + c2*sin(p*ζ))
    return uξ_B
end

function get_pq(S, T, ϵ, μ)
    r = (-1 + im*sqrt(3))/2
    c = sqrt(μ^2*T^2/4 + (1 + μ*S)^3/27)
    λ = 1/ϵ * sqrt(r*cbrt(-μ*T/2 + c) + conj(r)*cbrt(-μ*T/2 - c))
    q = real(λ)
    p = imag(λ)
    return p, q
end

function plot_pq()
    ϵ = 1e-3
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
        pq = get_pq.(Ss, T, ϵ, μ)
        ax[1].semilogx(Ss, ϵ*last.(pq),  c=c, ls=ls[i], label=label)
        ax[2].semilogx(Ss, ϵ*first.(pq), c=c, ls=ls[i], label=label)
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

# testrun()
plot_pq()
