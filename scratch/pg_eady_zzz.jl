using nuPGCM
using LinearAlgebra
using SparseArrays
using PyPlot
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function basic_state(z, Ri)
    nz = length(z)
    U = copy(z)
    V = zeros(nz)
    Bz = Ri*ones(nz)
    Uz = ones(nz)
    Vz = zeros(nz)
    return U, V, Bz, Uz, Vz
end

function growth_rate(eigenvals)
    isort = sortperm(imag(eigenvals), rev=true)
    for i ∈ eachindex(isort)
        if !isnan(eigenvals[isort[i]])
            return isort[i], imag(eigenvals[isort[i]])
        end
    end
end

function build_A_B!(A, B, k, l, z, μ, ε², U, V, Bz, Uz, Vz)
    nz = length(z)
    N = 4*nz # u, v, w, b
    imap = reshape(1:N, (4, nz)) 
    umap = imap[1, :]
    vmap = imap[2, :]
    wmap = imap[3, :]
    bmap = imap[4, :]
    A[:] .= 0
    B[:] .= 0
    for i=2:nz-1
        fd_z   = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz  = mkfdstencil(z[i-1:i+1], z[i], 2)
        if i == 2
            zzz_idxs = i-1:i+3
        elseif i == nz-1
            zzz_idxs = i-3:i+1
        else
            zzz_idxs = i-2:i+2
        end
        fd_zzz = mkfdstencil(z[zzz_idxs], z[i], 3)

        eq = imap[1, i]
        A[eq, vmap[i-1:i+1]] .+= -fd_z
        A[eq, bmap[i]] += im*k
        A[eq, umap[zzz_idxs]] .+= -ε²*fd_zzz
        
        eq = imap[2, i]
        A[eq, umap[i-1:i+1]] .+= fd_z
        A[eq, bmap[i]] += im*l
        A[eq, vmap[zzz_idxs]] .+= -ε²*fd_zzz

        eq = imap[3, i]
        A[eq, umap[i]] += im*k
        A[eq, vmap[i]] += im*l
        A[eq, wmap[i-1:i+1]] .+= fd_z

        eq = imap[4, i]
        A[eq, bmap[i]] += im*k*U[i] + im*l*V[i]
        A[eq, vmap[i]] += -1
        A[eq, wmap[i]] += Bz[i]
        A[eq, bmap[i-1:i+1]] .+= -ε²/μ*fd_zz
        B[eq, bmap[i]] += im
    end

    fd_z = mkfdstencil(z[1:3], z[1], 1)
    A[imap[1, 1], umap[1]] += 1
    A[imap[2, 1], vmap[1]] += 1
    A[imap[3, 1], wmap[1]] += 1
    A[imap[4, 1], bmap[1:3]] .+= fd_z

    fd_z  = mkfdstencil(z[nz-2:nz], z[nz], 1)
    A[imap[1, nz], umap[nz]] += 1
    A[imap[2, nz], vmap[nz]] += 1
    A[imap[3, nz], wmap[nz]] += 1
    A[imap[4, nz], bmap[nz-2:nz]] .+= fd_z

    return A, B
end
function build_A_B(k, l, z, μ, ε², U, V, Bz, Uz, Vz)
    N = 4*length(z)
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    return build_A_B!(A, B, k, l, z, μ, ε², U, V, Bz, Uz, Vz)
end

function compute_σ_grid(z, μ, ε², Ri)
    U, V, Bz, Uz, Vz = basic_state(z, Ri)
    n = 2^4
    kmax = 3
    lmax = 10
    k = 0:kmax/(n-1):kmax
    l = -lmax:2lmax/(n-1):lmax
    σ = zeros(length(l), length(k))
    N = 5*length(z)
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    @showprogress for i ∈ eachindex(k)
        for j ∈ eachindex(l)
            build_A_B!(A, B, k[i], l[j], z, μ, ε², U, V, Bz, Uz, Vz)
            F = eigen(A, B)
            idx, σ[j, i] = growth_rate(F.values)
        end
    end
    plot_σ_grid(k, l, σ)
    return k, l, σ
end

function plot_σ_grid(k, l, σ)
    fig, ax = plt.subplots(1)
    vmax = maximum(σ)
    extend = "neither"
    im = ax.pcolormesh(k, l, σ, vmin=0, vmax=vmax, shading="gouraud", rasterized=true, cmap="Reds")
    levels = range(0, maximum(σ), 4)
    ax.contour(k, l, σ, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    plt.colorbar(im, ax=ax, label="Growth rate", extend=extend)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xlabel(L"k")
    ax.set_ylabel(L"l")
    savefig("images/growth_rate.png")
    println("images/growth_rate.png")
    plt.close()
end

function plot_unstable_mode(k, l, z, μ, ε², Ri)
    # compute 
    U, V, Bz, Uz, Vz = basic_state(z, Ri)
    A, B = build_A_B(k, l, z, μ, ε², U, V, Bz, Uz, Vz)
    println(cond(A))
    println(cond(B))
    F = eigen(A, B)
    idx, σ = growth_rate(F.values)
    vec = F.vectors[:, idx]

    # unpack
    nz = length(z)
    N = 4*nz # u, v, w, b
    imap = reshape(1:N, (4, nz)) 
    umap = imap[1, :]
    vmap = imap[2, :]
    wmap = imap[3, :]
    bmap = imap[4, :]
    u = vec[umap]
    v = vec[vmap]
    w = vec[wmap]
    b = vec[bmap]

    # print energy limit
    σₑ = energy_constraint(u, v, w, b, Bz, z, ε²)
    @printf("            σ  = %1.1e\n", σ)

    # plot
    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2), sharey=true)
    ax[1].plot(real(u), z)
    ax[2].plot(real(v), z)
    ax[3].plot(real(w), z)
    ax[4].plot(real(b), z)
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"Re$[\hat u]$")
    ax[2].set_xlabel(L"Re$[\hat v]$")
    ax[3].set_xlabel(L"Re$[\hat w]$")
    ax[4].set_xlabel(L"Re$[\hat b]$")
    ax[2].set_title(string(L"\varepsilon^2 = ", @sprintf("%1.1e", ε²), L", \quad k = ", k, L", \quad l = ", l, L", $\quad$ Im$[\omega] = $", @sprintf("%1.1e", σ)))
    ax[1].set_yticks(-0.5:0.25:0.5)
    savefig("images/mum1D.png")
    println("images/mum1D.png")
    plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(8, 2.5), sharey=true)
    nx = 2^8
    x = -π/k:2π/k/(nx-1):π/k
    xx = repeat(x, 1, nz)
    zz = repeat(z', nx, 1)
    uu = repeat(u', nx, 1)
    vv = repeat(v', nx, 1)
    ww = repeat(w', nx, 1)
    bb = repeat(b', nx, 1)
    uu = @. uu*exp(im*k*xx)
    vv = @. vv*exp(im*k*xx)
    ww = @. ww*exp(im*k*xx)
    bb = @. bb*exp(im*k*xx)
    ax[1].set_ylabel(L"z")
    xz_plot(xx, zz, real(uu), ax[1], L"Re[$\hat u$]")
    xz_plot(xx, zz, real(vv), ax[2], L"Re[$\hat v$]")
    xz_plot(xx, zz, real(ww), ax[3], L"Re[$\hat w$]")
    xz_plot(xx, zz, real(bb), ax[4], L"Re[$\hat b$]")
    ax[2].set_title(string(L"\varepsilon^2 = ", @sprintf("%1.1e", ε²), L", \quad k = ", k, L", \quad l = ", l, L", $\quad$ Im$[\omega] = $", @sprintf("%1.1e", σ)))
    savefig("images/mum2D.png")
    println("images/mum2D.png")
    plt.close()
end

function energy_constraint(u, v, w, b, Bz, z, ε²)
    term1 = trapz(v.*conj(b) + conj(v).*b, z)
    term2a = trapz(Bz.*(w.*conj(b) + conj(w).*b), z)
    term2b = 2*ε²*trapz(Bz.*(abs.(differentiate(u, z)).^2 + abs.(differentiate(v, z)).^2), z)
    term3 = 2*ε²*trapz(abs.(differentiate(b, z)).^2, z)
    term4 = 2*trapz(abs.(b).^2, z)
    @printf("     vb* + v*b = %1.1e\n", term1/term4)
    @printf("Bz*(wb* + w*b) = %1.1e [≈ 2ε²Bz*(|∂z(u)|² + |∂z(v)|²) = %1.1e]\n", term2a/term4, term2b/term4)
    @printf("   2ε²|∂z(b)|² = %1.1e\n", term3/term4)
    @printf("         2|b|² = %0.1e\n", term4)
    σₑ = (term1 - term2a - term3)/term4
    @printf("            σₑ = %1.1e\n", σₑ)
    return σₑ
end

function xz_plot(xx, zz, uu, ax, label)
    vmax = maximum(abs.(uu))
    img = ax.pcolormesh(xx, zz, uu, vmin=-vmax, vmax=vmax, cmap="RdBu_r", shading="gouraud", rasterized=true)
    levels = range(-vmax, vmax, 8)
    ax.contour(xx, zz, uu, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    plt.colorbar(img, ax=ax, label=label, orientation="horizontal", pad=0.22)
    ax.set_xlabel(L"x")
    ax.set_xticks([minimum(xx), 0, maximum(xx)])
    ax.set_xticklabels([L"-\pi/k", L"0", L"\pi/k"])
    ax.set_yticks(-0.5:0.25:0.5)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
end

function main()
    nz = 2^6
    # z = -0.5:1/(nz-1):0.5
    z = -cos.(π*(0:nz-1)/(nz-1))/2
    μ = 1
    ε² = 1e-2
    Ri = 1

    # plot_basic_state(z, Ri)

    # k, l, σ = compute_σ_grid(z, μ, ε², Ri)

    plot_unstable_mode(0.5, -1, z, μ, ε², Ri)

    return
end

main()