using nuPGCM
using LinearAlgebra
using SparseArrays
using PyPlot
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# function basic_state(z, μ, ε²)
#     nz = length(z)
#     N = 3*nz # U, V, Bz
#     imap = reshape(1:N, (3, nz)) 
#     Umap = imap[1, :]
#     Vmap = imap[2, :]
#     Bzmap = imap[3, :]
#     A = Tuple{Int64,Int64,Float64}[]
#     r = zeros(N)
#     for i=2:nz-1
#         fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
#         fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)

#         eq = imap[1, i]
#         push!(A, (eq, Umap[i-1], -ε²*fd_zz[1]))
#         push!(A, (eq, Umap[i],   -ε²*fd_zz[2]))
#         push!(A, (eq, Umap[i+1], -ε²*fd_zz[3]))
#         push!(A, (eq, Vmap[i],   -1))
        
#         eq = imap[2, i]
#         push!(A, (eq, Vmap[i-1], -ε²*fd_zz[1]))
#         push!(A, (eq, Vmap[i],   -ε²*fd_zz[2]))
#         push!(A, (eq, Vmap[i+1], -ε²*fd_zz[3]))
#         push!(A, (eq, Umap[i],    1))
#         r[eq] = z[i]

#         eq = imap[3, i]
#         push!(A, (eq, Bzmap[i-1], -ε²/μ*fd_z[1]))
#         push!(A, (eq, Bzmap[i],   -ε²/μ*fd_z[2]))
#         push!(A, (eq, Bzmap[i+1], -ε²/μ*fd_z[3]))
#         push!(A, (eq, Vmap[i],    -1))
#     end

#     fd_z = mkfdstencil(z[1:3], z[1], 1)

#     eq = imap[1, 1]
#     push!(A, (eq, Umap[1], fd_z[1]))
#     push!(A, (eq, Umap[2], fd_z[2]))
#     push!(A, (eq, Umap[3], fd_z[3]))

#     eq = imap[2, 1]
#     push!(A, (eq, Vmap[1], fd_z[1]))
#     push!(A, (eq, Vmap[2], fd_z[2]))
#     push!(A, (eq, Vmap[3], fd_z[3]))

#     eq = imap[3, 1]
#     push!(A, (eq, Bzmap[1], 1))

#     fd_z  = mkfdstencil(z[nz-2:nz], z[nz], 1)

#     eq = imap[1, nz]
#     push!(A, (eq, Umap[nz-2], fd_z[1]))
#     push!(A, (eq, Umap[nz-1], fd_z[2]))
#     push!(A, (eq, Umap[nz],   fd_z[3]))

#     eq = imap[2, nz]
#     push!(A, (eq, Vmap[nz-2], fd_z[1]))
#     push!(A, (eq, Vmap[nz-1], fd_z[2]))
#     push!(A, (eq, Vmap[nz],   fd_z[3]))

#     eq = imap[3, nz]
#     push!(A, (eq, Bzmap[nz], 1))

#     A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
#     sol = A\r
#     U = sol[Umap]
#     V = sol[Vmap]
#     Bz = sol[Bzmap]
#     Uz = differentiate(U, z)
#     Vz = differentiate(V, z)
#     return U, V, Bz, Uz, Vz
# end

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
            # return isort[i+1], imag(eigenvals[isort[i+1]]) # take second biggest one because the first is constant mode
            return isort[i], imag(eigenvals[isort[i]])
        end
    end
end

function build_A_B!(A, B, k, l, z, μ, ε², U, V, Bz, Uz, Vz; primitive=false)
    nz = length(z)
    N = 5*nz # u, v, w, p, b
    imap = reshape(1:N, (5, nz)) 
    umap = imap[1, :]
    vmap = imap[2, :]
    wmap = imap[3, :]
    pmap = imap[4, :]
    bmap = imap[5, :]
    A[:] .= 0
    B[:] .= 0
    for i=2:nz-1
        fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)

        eq = imap[1, i]
        if primitive 
            A[eq, umap[i]] += im*k*U[i] + im*l*V[i]
            A[eq, wmap[i]] += Uz[i]
            B[eq, umap[i]] += im
        end
        A[eq, vmap[i]] += -1
        A[eq, pmap[i]] += im*k
        A[eq, umap[i-1]] += -ε²*fd_zz[1]
        A[eq, umap[i]]   += -ε²*fd_zz[2]
        A[eq, umap[i+1]] += -ε²*fd_zz[3]
        
        eq = imap[2, i]
        if primitive 
            A[eq, vmap[i]] += im*k*U[i] + im*l*V[i]
            A[eq, wmap[i]] += Vz[i]
            B[eq, vmap[i]] += im
        end
        A[eq, umap[i]] += 1
        A[eq, pmap[i]] += im*l
        A[eq, vmap[i-1]] += -ε²*fd_zz[1]
        A[eq, vmap[i]]   += -ε²*fd_zz[2]
        A[eq, vmap[i+1]] += -ε²*fd_zz[3]

        eq = imap[3, i]
        A[eq, bmap[i]] += 1
        A[eq, pmap[i-1]] += -fd_z[1]
        A[eq, pmap[i]]   += -fd_z[2]
        A[eq, pmap[i+1]] += -fd_z[3]

        eq = imap[4, i]
        A[eq, umap[i]] += im*k
        A[eq, vmap[i]] += im*l
        A[eq, wmap[i-1]] += fd_z[1]
        A[eq, wmap[i]]   += fd_z[2]
        A[eq, wmap[i+1]] += fd_z[3]

        eq = imap[5, i]
        A[eq, bmap[i]] += im*k*U[i] + im*l*V[i]
        A[eq, vmap[i]] += -1
        A[eq, wmap[i]] += Bz[i]
        A[eq, bmap[i-1]] += -ε²/μ*fd_zz[1]
        A[eq, bmap[i]]   += -ε²/μ*fd_zz[2]
        A[eq, bmap[i+1]] += -ε²/μ*fd_zz[3]
        B[eq, bmap[i]] += im
    end

    fd_z = mkfdstencil(z[1:3], z[1], 1)

    eq = imap[1, 1]
    A[eq, umap[1]] += 1

    eq = imap[2, 1]
    # A[eq, vmap[1]] += 1
    A[eq, vmap[1]] += fd_z[1]
    A[eq, vmap[2]] += fd_z[2]
    A[eq, vmap[3]] += fd_z[3]

    eq = imap[3, 1]
    A[eq, pmap[1]] += -fd_z[1]
    A[eq, pmap[2]] += -fd_z[2]
    A[eq, pmap[3]] += -fd_z[3]
    A[eq, bmap[1]] += 1

    eq = imap[4, 1]
    A[eq, wmap[1]] += 1

    eq = imap[5, 1]
    A[eq, bmap[1]] += fd_z[1]
    A[eq, bmap[2]] += fd_z[2]
    A[eq, bmap[3]] += fd_z[3]

    fd_z  = mkfdstencil(z[nz-2:nz], z[nz], 1)

    eq = imap[1, nz]
    A[eq, umap[nz]] += 1

    eq = imap[2, nz]
    # A[eq, vmap[nz]] += 1
    A[eq, vmap[nz-2]] += fd_z[1]
    A[eq, vmap[nz-1]] += fd_z[2]
    A[eq, vmap[nz]]   += fd_z[3]

    eq = imap[3, nz]
    A[eq, pmap[nz-2]] += -fd_z[1]
    A[eq, pmap[nz-1]] += -fd_z[2]
    A[eq, pmap[nz]]   += -fd_z[3]
    A[eq, bmap[nz]] += 1

    eq = imap[4, nz]
    A[eq, wmap[nz]] += 1

    eq = imap[5, nz]
    A[eq, bmap[nz-2]] += fd_z[1]
    A[eq, bmap[nz-1]] += fd_z[2]
    A[eq, bmap[nz]]   += fd_z[3]

    return A, B
end
function build_A_B(k, l, z, μ, ε², U, V, Bz, Uz, Vz; primitive=false)
    N = 5*length(z)
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    return build_A_B!(A, B, k, l, z, μ, ε², U, V, Bz, Uz, Vz; primitive=primitive)
end

function plot_basic_state(z, Ri)
    fig, ax = plt.subplots(1, 3, figsize=(6, 3.2), sharey=true)
    for ε² = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        U, V, Bz, Uz, Vz = basic_state(z, Ri)
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

function compute_σ_grid(z, μ, ε², Ri; primitive=false)
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
            build_A_B!(A, B, k[i], l[j], z, μ, ε², U, V, Bz, Uz, Vz, primitive=primitive)
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

function plot_unstable_mode(k, l, z, μ, ε², Ri; primitive=false)
    # compute 
    U, V, Bz, Uz, Vz = basic_state(z, Ri)
    A, B = build_A_B(k, l, z, μ, ε², U, V, Bz, Uz, Vz, primitive=primitive)
    println(cond(A))
    println(cond(B))
    F = eigen(A, B)
    idx, σ = growth_rate(F.values)
    vec = F.vectors[:, idx]

    # unpack
    nz = length(z)
    N = 5*nz # u, v, w, p, b
    imap = reshape(1:N, (5, nz)) 
    umap = imap[1, :]
    vmap = imap[2, :]
    wmap = imap[3, :]
    pmap = imap[4, :]
    bmap = imap[5, :]
    u = vec[umap]
    v = vec[vmap]
    w = vec[wmap]
    p = vec[pmap]
    b = vec[bmap]

    # print energy limit
    σₑ = energy_constraint(u, v, w, b, Bz, z, ε²)
    @printf("            σ  = %1.1e\n", σ)

    # plot
    fig, ax = plt.subplots(1, 5, figsize=(10, 3.2), sharey=true)
    ax[1].plot(real(u), z)
    ax[2].plot(real(v), z)
    ax[3].plot(real(w), z)
    ax[4].plot(real(p), z)
    ax[5].plot(real(b), z)
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"Re$[\hat u]$")
    ax[2].set_xlabel(L"Re$[\hat v]$")
    ax[3].set_xlabel(L"Re$[\hat w]$")
    ax[4].set_xlabel(L"Re$[\hat p]$")
    ax[5].set_xlabel(L"Re$[\hat b]$")
    ax[3].set_title(string(L"\varepsilon^2 = ", @sprintf("%1.1e", ε²), L", \quad k = ", k, L", \quad l = ", l, L", $\quad$ Im$[\omega] = $", @sprintf("%1.1e", σ)))
    ax[1].set_yticks(-0.5:0.25:0.5)
    savefig("images/mum1D.png")
    println("images/mum1D.png")
    plt.close()

    fig, ax = plt.subplots(1, 5, figsize=(10, 2.5), sharey=true)
    nx = 2^8
    x = -π/k:2π/k/(nx-1):π/k
    xx = repeat(x, 1, nz)
    zz = repeat(z', nx, 1)
    uu = repeat(u', nx, 1)
    vv = repeat(v', nx, 1)
    ww = repeat(w', nx, 1)
    pp = repeat(p', nx, 1)
    bb = repeat(b', nx, 1)
    uu = @. uu*exp(im*k*xx)
    vv = @. vv*exp(im*k*xx)
    ww = @. ww*exp(im*k*xx)
    pp = @. pp*exp(im*k*xx)
    bb = @. bb*exp(im*k*xx)
    ax[1].set_ylabel(L"z")
    xz_plot(xx, zz, real(uu), ax[1], L"Re[$\hat u$]")
    xz_plot(xx, zz, real(vv), ax[2], L"Re[$\hat v$]")
    xz_plot(xx, zz, real(ww), ax[3], L"Re[$\hat w$]")
    xz_plot(xx, zz, real(pp), ax[4], L"Re[$\hat p$]")
    xz_plot(xx, zz, real(bb), ax[5], L"Re[$\hat b$]")
    ax[3].set_title(string(L"\varepsilon^2 = ", @sprintf("%1.1e", ε²), L", \quad k = ", k, L", \quad l = ", l, L", $\quad$ Im$[\omega] = $", @sprintf("%1.1e", σ)))
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

    # k, l, σ = compute_σ_grid(z, μ, ε², Ri, primitive=false)

    plot_unstable_mode(4.1, 0, z, μ, ε², Ri, primitive=false)

    return
end

main()