using nuPGCM
using LinearAlgebra
using SparseArrays
using PyPlot
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function nanmax(a)
    return maximum(x -> isnan(x) ? -Inf : x, a)
end

function build_A_B(k, l, z, μ, ε², U, V, dUdz, dVdz, dBdz)
    nz = length(z)
    N = 5*nz # u, v, w, p, b
    imap = reshape(1:N, (5, nz)) 
    umap = imap[1, :]
    vmap = imap[2, :]
    wmap = imap[3, :]
    pmap = imap[4, :]
    bmap = imap[5, :]
    A = Tuple{Int64,Int64,ComplexF64}[]
    B = Tuple{Int64,Int64,ComplexF64}[]
    for i=2:nz-1
        fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)

        eq = imap[1, i]
        push!(A, (eq, umap[i], im*k*U[i] + im*l*V[i]))
        push!(A, (eq, wmap[i], dUdz[i]))
        push!(A, (eq, vmap[i], -1))
        push!(A, (eq, pmap[i], im*k))
        push!(A, (eq, umap[i-1], -ε²*fd_zz[1]))
        push!(A, (eq, umap[i],   -ε²*fd_zz[2]))
        push!(A, (eq, umap[i+1], -ε²*fd_zz[3]))
        push!(B, (eq, umap[i], im))
        
        eq = imap[2, i]
        push!(A, (eq, vmap[i], im*k*U[i] + im*l*V[i]))
        push!(A, (eq, wmap[i], dVdz[i]))
        push!(A, (eq, umap[i], 1))
        push!(A, (eq, pmap[i], im*l))
        push!(A, (eq, vmap[i-1], -ε²*fd_zz[1]))
        push!(A, (eq, vmap[i],   -ε²*fd_zz[2]))
        push!(A, (eq, vmap[i+1], -ε²*fd_zz[3]))
        push!(B, (eq, vmap[i], im))

        eq = imap[3, i]
        push!(A, (eq, bmap[i], 1))
        push!(A, (eq, pmap[i-1], -fd_z[1]))
        push!(A, (eq, pmap[i],   -fd_z[2]))
        push!(A, (eq, pmap[i+1], -fd_z[3]))

        eq = imap[4, i]
        push!(A, (eq, umap[i], im*k))
        push!(A, (eq, vmap[i], im*l))
        push!(A, (eq, wmap[i-1], fd_z[1]))
        push!(A, (eq, wmap[i],   fd_z[2]))
        push!(A, (eq, wmap[i+1], fd_z[3]))

        eq = imap[5, i]
        push!(A, (eq, bmap[i], im*k*U[i] + im*l*V[i]))
        push!(A, (eq, vmap[i], -1))
        push!(A, (eq, wmap[i], dBdz[i]))
        push!(A, (eq, bmap[i-1], -ε²/μ*fd_zz[1]))
        push!(A, (eq, bmap[i],   -ε²/μ*fd_zz[2]))
        push!(A, (eq, bmap[i+1], -ε²/μ*fd_zz[3]))
        push!(B, (eq, bmap[i], im))
    end

    fd_z = mkfdstencil(z[1:3], z[1], 1)

    eq = imap[1, 1]
    push!(A, (eq, umap[1], fd_z[1]))
    push!(A, (eq, umap[2], fd_z[2]))
    push!(A, (eq, umap[3], fd_z[3]))

    eq = imap[2, 1]
    push!(A, (eq, vmap[1], fd_z[1]))
    push!(A, (eq, vmap[2], fd_z[2]))
    push!(A, (eq, vmap[3], fd_z[3]))

    eq = imap[3, 1]
    push!(A, (eq, pmap[1], 1))

    eq = imap[4, 1]
    push!(A, (eq, wmap[1], 1))

    eq = imap[5, 1]
    push!(A, (eq, bmap[1], fd_z[1]))
    push!(A, (eq, bmap[2], fd_z[2]))
    push!(A, (eq, bmap[3], fd_z[3]))

    fd_z  = mkfdstencil(z[nz-2:nz], z[nz], 1)

    eq = imap[1, nz]
    push!(A, (eq, umap[nz-2], fd_z[1]))
    push!(A, (eq, umap[nz-1], fd_z[2]))
    push!(A, (eq, umap[nz],   fd_z[3]))

    eq = imap[2, nz]
    push!(A, (eq, vmap[nz-2], fd_z[1]))
    push!(A, (eq, vmap[nz-1], fd_z[2]))
    push!(A, (eq, vmap[nz],   fd_z[3]))

    eq = imap[3, nz]
    push!(A, (eq, pmap[nz-2], -fd_z[1]))
    push!(A, (eq, pmap[nz-1], -fd_z[2]))
    push!(A, (eq, pmap[nz],   -fd_z[3]))
    push!(A, (eq, bmap[nz], 1))

    eq = imap[4, nz]
    push!(A, (eq, wmap[nz], 1))

    eq = imap[5, nz]
    push!(A, (eq, bmap[nz-2], fd_z[1]))
    push!(A, (eq, bmap[nz-1], fd_z[2]))
    push!(A, (eq, bmap[nz],   fd_z[3]))

    return sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N), sparse((x->x[1]).(B), (x->x[2]).(B), (x->x[3]).(B), N, N)
end

function basic_state(z, μ, ε²)
    q = 1/sqrt(ε²)
    # U = cumtrapz(dUdz, z) .- trapz(dUdz, z)/2
    # V = -ε²*differentiate(dUdz, z)
    # dVdz = differentiate(V, z)
    # dBdz = -μ/ε²*cumtrapz(V, z)
    dUdz = @. 1 - 2*(cosh(q/2)*cos(q/2)*cosh(q*z)*cos(q*z) + sinh(q/2)*sin(q/2)*sinh(q*z)*sin(q*z))/(cosh(q) + cos(q))
    U = @. (q*z*cos(q) + q*z*cosh(q) - cosh(q*z)*sin(q/2)*sin(q*z)*sinh(q/2) + cos(q*z)*sin(q/2)*sinh(q/2)*sinh(q*z) - cos(q/2)*cosh(q/2)*(cosh(q*z)*sin(q*z) + cos(q*z)*sinh(q*z)))/(q*(cos(q)+cosh(q)))
    V = @. (2*(cos(q/2)*cosh(q/2)*(-cosh(q*z)*sin(q*z) + cos(q*z)*sinh(q*z)) + sin(q/2)*sinh(q/2)*(cosh(q*z)*sin(q*z) + cos(q*z)*sinh(q*z))))/(q*(cos(q) + cosh(q)))
    dVdz = @. (4*cos(q*z)*cosh(q*z)*sin(q/2)*sinh(q/2)-4*cos(q/2)*cosh(q/2)*sin(q*z)*sinh(q*z))/(cos(q)+cosh(q))
    dBdz = @. -((2*μ*(cos(q/2)*cos(q*z)*cosh(q/2)*cosh(q*z)+sin(q/2)*sin(q*z)*sinh(q/2)*sinh(q*z)))/(cos(q)+cosh(q)))
    return U, V, dUdz, dVdz, dBdz
end

function plot_basic_state(z, μ)
    fig, ax = plt.subplots(1, 3, figsize=(6, 3.2), sharey=true)
    for ε² = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        U, V, dBdz = basic_state(z, μ, ε²)
        ax[1].plot(U, z, label=ε²)
        ax[2].plot(V, z)
        ax[3].plot(dBdz, z)
    end
    ax[1].legend()
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"\bar u")
    ax[2].set_xlabel(L"\bar v")
    ax[3].set_xlabel(L"\partial_z \bar b")
    savefig("images/basic_state.png")
    println("images/basic_state.png")
    plt.close()
end

function compute_omega(z, μ, ε²)
    U, V, dUdz, dVdz, dBdz = basic_state(z, μ, ε²)
    n = 10
    kmax = 100
    lmax = 100
    k = 0:kmax/(n-1):kmax
    l = -lmax:2lmax/(n-1):lmax
    ω = zeros(length(l), length(k))
    @showprogress for i ∈ eachindex(k)
        for j ∈ eachindex(l)
            A, B = build_A_B(k[i], l[j], z, μ, ε², U, V, dUdz, dVdz, dBdz)
            # F = eigen(Array(A), Array(B))
            # ω[j, i] = nanmax(imag(F.values))
            C = inv(Array(A))*Array(B)
            F = eigen(C)
            ω[j, i] = nanmax(imag(1 ./F.values))
        end
    end
    fig, ax = plt.subplots(1)
    im = ax.pcolormesh(k, l, ω, shading="gouraud", rasterized=true, cmap="Reds")
    ax.contour(k, l, ω, colors="k", linestyles="-", linewidths=0.25)
    plt.colorbar(im, ax=ax, label="Growth rate")
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
    ε² = 1e-1

    plot_basic_state(z, μ)

    compute_omega(z, μ, ε²)

    # U, V, dUdz, dVdz, dBdz = basic_state(z, μ, ε²)
    # A, B = build_A_B(1, 1, z, μ, ε², U, V, dUdz, dVdz, dBdz)
    # println(rank(A))
    # println(rank(B))
    # println(5*nz)

    return
end

main()