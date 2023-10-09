using nuPGCM
using LinearAlgebra
using SparseArrays
using PyPlot
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

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
    σ = sort(imag(eigenvals), rev=true)
    for i ∈ eachindex(σ)
        if !isnan(σ[i])
            return σ[i+1] # take second biggest one because the first is constant mode
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
    A[eq, umap[1]] += fd_z[1]
    A[eq, umap[2]] += fd_z[2]
    A[eq, umap[3]] += fd_z[3]

    eq = imap[2, 1]
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
    A[eq, umap[nz-2]] += fd_z[1]
    A[eq, umap[nz-1]] += fd_z[2]
    A[eq, umap[nz]]   += fd_z[3]

    eq = imap[2, nz]
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

function compute_omega(z, μ, ε²; primitive=false)
    U, V, Bz, Uz, Vz = basic_state(z, μ, ε²)
    n = 2^5
    kmax = 3
    lmax = 10
    k = 0:kmax/(n-1):kmax
    l = -lmax:2lmax/(n-1):lmax
    ω = zeros(length(l), length(k))
    N = 5*length(z)
    A = zeros(ComplexF64, N, N)
    B = zeros(ComplexF64, N, N)
    @showprogress for i ∈ eachindex(k)
        for j ∈ eachindex(l)
            A, B = build_A_B!(A, B, k[i], l[j], z, μ, ε², U, V, Bz, Uz, Vz, primitive=primitive)
            F = eigen(A, B)
            ω[j, i] = growth_rate(F.values)
        end
    end
    return k, l, ω
end

function plot_omega(k, l, ω)
    fig, ax = plt.subplots(1)
    if maximum(ω) > 1
        vmax = 1
        extend = "max"
    else
        vmax = maximum(ω)
        extend = "neither"
    end
    im = ax.pcolormesh(k, l, ω, vmin=0, vmax=vmax, shading="gouraud", rasterized=true, cmap="Reds")
    levels = range(0, maximum(ω), 4)
    ax.contour(k, l, ω, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    plt.colorbar(im, ax=ax, label="Growth rate", extend=extend)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xlabel(L"k")
    ax.set_ylabel(L"l")
    savefig("images/omega.png")
    println("images/omega.png")
    plt.close()
end

function main()
    nz = 2^5
    # z = -0.5:1/(nz-1):0.5
    z = -cos.(π*(0:nz-1)/(nz-1))/2
    μ = 1
    ε² = 1e-2

    plot_basic_state(z, μ)

    k, l, ω = compute_omega(z, μ, ε², primitive=false)
    plot_omega(k, l, ω)

    # U, V, dUdz, dVdz, dBdz = basic_state(z, μ, ε²)
    # A, B = build_A_B(2, -2, z, μ, ε², U, V, dUdz, dVdz, dBdz)
    # F = eigen(Array(A), Array(B))
    # ω = nanmax(imag(F.values))
    # a = sort(imag(F.values), rev=true)
    # for i ∈ a
    #     println(i)
    # end
    # display(ω)
    # println(rank(A))
    # println(rank(B))
    # println(5*nz)

    return
end

main()