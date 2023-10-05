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

function build_A_B(k, l, z)
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

        eq = imap[1, i]
        push!(A, (eq, umap[i], im*k*z[i]))
        push!(A, (eq, vmap[i], -1))
        push!(A, (eq, pmap[i], im*k))
        push!(B, (eq, umap[i], im))
        
        eq = imap[2, i]
        push!(A, (eq, vmap[i], im*k*z[i]))
        push!(A, (eq, umap[i], 1))
        push!(A, (eq, pmap[i], im*l))
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
        push!(A, (eq, bmap[i], im*k*z[i]))
        push!(A, (eq, vmap[i], -1))
        push!(A, (eq, wmap[i], 1))
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
    push!(A, (eq, vmap[1], -1))
    push!(B, (eq, bmap[1], im))

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
    push!(A, (eq, bmap[nz], im*k))
    push!(A, (eq, vmap[nz], -1))
    push!(B, (eq, bmap[nz], im))

    return sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N), sparse((x->x[1]).(B), (x->x[2]).(B), (x->x[3]).(B), N, N)
end

function compute_omega(z)
    k = 0:0.1:4
    ω = zeros(length(k))
    @showprogress for i ∈ eachindex(k)
        A, B = build_A_B(k[i], 0, z)
        F = eigen(Array(A), Array(B))
        ω[i] = nanmax(imag(F.values))
    end
    fig, ax = plt.subplots(1)
    im = ax.plot(k, ω)
    ax.set_xlabel(L"k")
    ax.set_ylabel(L"\omega")
    ax.set_ylim(0, 1)
    savefig("images/omega.png")
    println("images/omega.png")
    plt.close()
end

function main()
    nz = 2^5
    z = 0:1/(nz-1):1

    compute_omega(z)

    return
end

main()