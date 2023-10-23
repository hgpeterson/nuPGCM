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
    Bz = sqrt(Ri)*ones(nz)
    Uz = ones(nz)
    Vz = zeros(nz)
    return U, V, Bz, Uz, Vz
end

function basic_state(z, μ, ε²)
    nz = length(z)
    N = 3*nz # U, V, Bz
    imap = reshape(1:N, (3, nz)) 
    Umap = imap[1, :]
    Vmap = imap[2, :]
    Bzmap = imap[3, :]
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for i=2:nz-1
        fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)

        eq = imap[1, i]
        push!(A, (eq, Umap[i-1], -ε²*fd_zz[1]))
        push!(A, (eq, Umap[i],   -ε²*fd_zz[2]))
        push!(A, (eq, Umap[i+1], -ε²*fd_zz[3]))
        push!(A, (eq, Vmap[i],   -1))
        
        eq = imap[2, i]
        push!(A, (eq, Vmap[i-1], -ε²*fd_zz[1]))
        push!(A, (eq, Vmap[i],   -ε²*fd_zz[2]))
        push!(A, (eq, Vmap[i+1], -ε²*fd_zz[3]))
        push!(A, (eq, Umap[i],    1))
        r[eq] = z[i]

        eq = imap[3, i]
        push!(A, (eq, Bzmap[i-1], -ε²/μ*fd_z[1]))
        push!(A, (eq, Bzmap[i],   -ε²/μ*fd_z[2]))
        push!(A, (eq, Bzmap[i+1], -ε²/μ*fd_z[3]))
        push!(A, (eq, Vmap[i],    -1))
    end

    fd_z = mkfdstencil(z[1:3], z[1], 1)

    eq = imap[1, 1]
    push!(A, (eq, Umap[1], fd_z[1]))
    push!(A, (eq, Umap[2], fd_z[2]))
    push!(A, (eq, Umap[3], fd_z[3]))

    eq = imap[2, 1]
    push!(A, (eq, Vmap[1], fd_z[1]))
    push!(A, (eq, Vmap[2], fd_z[2]))
    push!(A, (eq, Vmap[3], fd_z[3]))

    eq = imap[3, 1]
    push!(A, (eq, Bzmap[1], 1))

    fd_z  = mkfdstencil(z[nz-2:nz], z[nz], 1)

    eq = imap[1, nz]
    push!(A, (eq, Umap[nz-2], fd_z[1]))
    push!(A, (eq, Umap[nz-1], fd_z[2]))
    push!(A, (eq, Umap[nz],   fd_z[3]))

    eq = imap[2, nz]
    push!(A, (eq, Vmap[nz-2], fd_z[1]))
    push!(A, (eq, Vmap[nz-1], fd_z[2]))
    push!(A, (eq, Vmap[nz],   fd_z[3]))

    eq = imap[3, nz]
    push!(A, (eq, Bzmap[nz], 1))

    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    sol = A\r
    U = sol[Umap]
    V = sol[Vmap]
    Bz = sol[Bzmap]
    Uz = differentiate(U, z)
    Vz = differentiate(V, z)
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

function build_A_B!(A, B, k, l, z, μ, ε², U, V, Bz, Uz, Vz; bc="no slip")
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
        fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)
        if i == 2
            zzz_idxs = i:i+4
        elseif i == nz-1
            zzz_idxs = i-4:i
        else
            zzz_idxs = i-2:i+2
        end
        fd_zzz = mkfdstencil(z[zzz_idxs], z[i], 3)

        eq = imap[1, i]
        if primitive 
            A[eq, umap[i]] += im*k*Uz[i] + im*l*Vz[i]
            A[eq, umap[i-1:i+1]] .+= (im*k*U[i] + im*l*V[i])*fd_z
            A[eq, wmap[i]] += Uzz[i]
            A[eq, wmap[i-1:i+1]] .+= Uz[i]*fd_z
            B[eq, umap[i-1:i+1]] .+= im*fd_z
        end
        A[eq, vmap[i-1:i+1]] .+= -fd_z
        A[eq, bmap[i]] += im*k
        A[eq, umap[zzz_idxs]] .+= -ε²*fd_zzz
        
        eq = imap[2, i]
        if primitive 
            A[eq, vmap[i]] += im*k*Uz[i] + im*l*Vz[i]
            A[eq, vmap[i-1:i+1]] .+= (im*k*U[i] + im*l*V[i])*fd_z
            A[eq, wmap[i]] += Vzz[i]
            A[eq, wmap[i-1:i+1]] .+= Vz[i]*fd_z
            B[eq, vmap[i-1:i+1]] .+= im*fd_z
        end
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
    if bc == "no slip"
        A[imap[1, 1], umap[1]] += 1
        A[imap[2, 1], vmap[1]] += 1
    elseif bc == "free slip"
        A[imap[1, 1], umap[1:3]] .+= fd_z
        A[imap[2, 1], vmap[1:3]] .+= fd_z
    else 
        error()
    end
    A[imap[3, 1], wmap[1]] += 1
    A[imap[4, 1], bmap[1:3]] .+= fd_z

    fd_z  = mkfdstencil(z[nz-2:nz], z[nz], 1)
    if bc == "no slip"
        A[imap[1, nz], umap[nz]] += 1
        A[imap[2, nz], vmap[nz]] += 1
    elseif bc == "free slip"
        A[imap[1, nz], umap[nz-2:nz]] .+= fd_z
        A[imap[2, nz], vmap[nz-2:nz]] .+= fd_z
    end
    A[imap[3, nz], wmap[nz]] += 1
    A[imap[4, nz], bmap[nz-2:nz]] .+= fd_z

    return A, B
end
function build_A_B(k, l, z, μ, ε², U, V, Bz, Uz, Vz; bc)
    N = 4*length(z)
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    return build_A_B!(A, B, k, l, z, μ, ε², U, V, Bz, Uz, Vz; bc)
end

function compute_σ_grid(z, μ, ε², Ri; bc="no slip")
    if bc == "no slip"
        U, V, Bz, Uz, Vz = basic_state(z, Ri)
    elseif bc == "free slip"
        U, V, Bz, Uz, Vz = basic_state(z, μ, ε²)
    else
        error()
    end
    n = 2^4
    kmax = 1.5
    lmax = 5
    k = 0:kmax/(n-1):kmax
    l = -lmax:2lmax/(n-1):lmax
    σ = zeros(length(l), length(k))
    N = 4*length(z)
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    @showprogress for i ∈ eachindex(k)
        for j ∈ eachindex(l)
            build_A_B!(A, B, k[i], l[j], z, μ, ε², U, V, Bz, Uz, Vz; bc)
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

function plot_unstable_mode(k, l, z, μ, ε², Ri; bc="no slip")
    # compute 
    if bc == "no slip"
        U, V, Bz, Uz, Vz = basic_state(z, Ri)
    elseif bc == "free slip"
        U, V, Bz, Uz, Vz = basic_state(z, μ, ε²)
    else
        error()
    end
    A, B = build_A_B(k, l, z, μ, ε², U, V, Bz, Uz, Vz; bc)
    println("Cond(A) = ", cond(A))
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
    ax[2].set_title(string(L"\varepsilon^2 = ", @sprintf("%1.1e", ε²), L", \quad k = ", latexstring(k), L", \quad l = ", latexstring(l), L", $\quad$ Im$[\omega] = $", @sprintf("%1.1e", σ)))
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

function plot_basic_state(z, μ)
    fig, ax = plt.subplots(1, 3, figsize=(6, 3.2), sharey=true)
    for ε² = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        U, V, Bz, Uz, Vz = basic_state(z, μ, ε²)
        ax[1].plot(U, z, label=ε²)
        ax[2].plot(V, z)
        ax[3].plot(Bz, z)
    end
    ax[1].set_xlim(-0.6, 0.6)
    ax[1].set_xticks(-0.6:0.2:0.6)
    ax[2].set_xlim(-0.25, 0.25)
    ax[2].set_xticks(-0.2:0.1:0.2)
    ax[1].legend()
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"\bar u")
    ax[2].set_xlabel(L"\bar v")
    ax[3].set_xlabel(L"\partial_z \bar b")
    savefig("images/basic_state.png")
    println("images/basic_state.png")
    plt.close()
end

function main()
    nz = 2^5
    z = collect(-0.5:1/(nz-1):0.5)
    # z = -cos.(π*(0:nz-1)/(nz-1))/2
    μ = 1
    ε² = 1e-2
    Ri = 1
    bc = "no slip"
    # bc = "free slip"

    # plot_basic_state(z, μ)

    # k, l, σ = compute_σ_grid(z, μ, ε², Ri; bc)

    plot_unstable_mode(0.5, -2, z, μ, ε², Ri; bc)

    return
end

main()