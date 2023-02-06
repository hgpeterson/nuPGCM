using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
2D Baroclinic:
    -ε²∂zz(ωˣ) - ωʸ = 0,
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
BC:
    • ωˣ = 0 at z = 0
    • ωˣ = 0 at z = -H
    • ωˣ = Uˣ/ε² at z = 0
    • ∫ zωʸ dz = -Uˣ
"""
function solve_baroclinic_fd(z, bx, ε², Ux)
    # indices
    nz = size(z, 1)
    ωxmap = 1:nz
    ωymap = (nz+1):2*nz

    # matrix
    A = Tuple{Int64,Int64,Float64}[]  
    r = zeros(2*nz)

    # interior nodes
    for j=2:nz-1 
        # ∂zz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # eqtn 1: -ε²∂zz(ωˣ) - ωʸ = 0
        # term 1
        push!(A, (ωxmap[j], ωxmap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωxmap[j], ωxmap[j],   -ε²*fd_zz[2]))
        push!(A, (ωxmap[j], ωxmap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωxmap[j], ωymap[j], -1))

        # eqtn 2: -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
        # term 1
        push!(A, (ωymap[j], ωymap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωymap[j], ωymap[j],   -ε²*fd_zz[2]))
        push!(A, (ωymap[j], ωymap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωymap[j], ωxmap[j], 1))
        # rhs
        r[ωymap[j]] = -bx[j]
    end

    # ωˣ = ωʸ = 0 at z = 0
    push!(A, (ωxmap[nz], ωxmap[nz], 1))
    push!(A, (ωymap[nz], ωymap[nz], 1))

    # ωˣ = Uˣ/ε² at z = -H
    push!(A, (ωxmap[1], ωxmap[1], 1))
    r[ωxmap[1]] = Ux/ε²

    # ∫ zωʸ dz = -Ux
    for j=1:nz-1
        push!(A, (ωymap[1], ωymap[j],     z[j]*(z[j+1] - z[j])/2))
        push!(A, (ωymap[1], ωymap[j+1], z[j+1]*(z[j+1] - z[j])/2))
    end
    r[ωymap[1]] = -Ux

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), 2*nz, 2*nz)

    sol = A\r
    return sol[ωxmap], sol[ωymap]
end

function solve_baroclinic_fe(g, g1, ε², bx, Ux)
    # indices
    ωxmap = 1:g.np
    ωymap = (g.np+1):2*g.np

    # integrals and Jacobians
    s = ShapeFunctionIntegrals(g.s, g.s)
    J = Jacobians(g1)

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(2*g.np)
    for k=1:g.nt
        # stiffness and mass matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # RHS
        r[ωymap[g.t[k, :]]] -= M*bx[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ∈ g.e
                continue
            end

            # indices
            ωxi = ωxmap[g.t[k, :]]
            ωyi = ωymap[g.t[k, :]]

            # -ε²∂zz(ωx)
            push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
            # -ωy
            push!(A, (ωxi[i], ωyi[j], -M[i, j]))

            # -ε²∂zz(ωy)
            push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
            # +ωx
            push!(A, (ωyi[i], ωxi[j], M[i, j]))
        end
    end

    # ωˣ(0) = ωʸ(0) = 0
    push!(A, (ωxmap[g.e[2]], ωxmap[g.e[2]], 1))
    push!(A, (ωymap[g.e[2]], ωymap[g.e[2]], 1))

    # ωˣ(-H) = Uˣ/ε²
    push!(A, (ωxmap[g.e[1]], ωxmap[g.e[1]], 1))
    r[ωxmap[1]] = Ux/ε²

    # ∫ zωʸ dz = -Ux
    w, ξ = quad_weights_points(g.order+1, 1)
    for k=1:g.nt
        for i=1:g.nn
            func(ξ) = transform_from_ref_el(ξ, g1.p[g1.t[k, :], :])*φ(g.s, i, ξ)*J.dets[k]
            push!(A, (ωymap[g.e[1]], ωymap[g.t[k, i]], nuPGCM.ref_el_quad(func, w, ξ)))
        end
    end
    r[ωymap[g.e[1]]] = -Ux

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), 2*g.np, 2*g.np)

    # remove zeros
    dropzeros!(A)

    # solve
    sol = A\r

    # reshape 
    ωx = FEField(sol[ωxmap], g, g1)
    ωy = FEField(sol[ωymap], g, g1)
    return ωx, ωy
end

function get_u(ωx, ωy, z)
    # uˣ, uʸ
    return cumtrapz(ωy, z), -cumtrapz(ωx, z)
end

function baroclinic_fd_fe(; order)
    # solve fd at high res
    ωx_fd, ωy_fd = solve_baroclinic_fd(z_hr, bx.(z_hr), ε², Ux)

    # FE grid
    p = reshape(z, (nz, 1))
    t = hcat(1:nz-1, 2:nz)
    e = [1, nz]
    g = FEGrid(p, t, e, order)
    g1 = FEGrid(p, t, e, 1)

    # solve fe
    ωx_fe, ωy_fe = solve_baroclinic_fe(g, g1, ε², bx.(g.p), Ux)

    # plot
    fig, ax = subplots(1, figsize=(2, 3.2))
    perm = sortperm(g.p[:, 1])
    ax.plot(ωx_fd, z_hr, label=L"$\omega^x$")
    ax.plot(ωy_fd, z_hr, label=L"$\omega^y$")
    ax.plot(ωx_fe.values[perm], g.p[perm], "k--", lw=0.5, label="FE")
    ax.plot(ωy_fe.values[perm], g.p[perm], "k--", lw=0.5)
    ax.legend()
    ax.set_xlabel(L"\omega")
    ax.set_ylabel(L"z")
    savefig("scratch/images/omega.png")
    println("scratch/images/omega.png")
    plt.close()

    # velocities
    ux_fd, uy_fd = get_u(ωx_fd, ωy_fd, z_hr)
    ux_fe, uy_fe = get_u(ωx_fe.values[perm], ωy_fe.values[perm], g.p[perm])

    # plot
    fig, ax = subplots(1, figsize=(2, 3.2))
    ax.plot(ux_fd, z_hr, label=L"$u^x$")
    ax.plot(uy_fd, z_hr, label=L"$u^y$")
    ax.plot(ux_fe, g.p[perm], "k--", lw=0.5, label="FE")
    ax.plot(uy_fe, g.p[perm], "k--", lw=0.5)
    ax.legend()
    ax.set_xlabel(L"u")
    ax.set_ylabel(L"z")
    savefig("scratch/images/u.png")
    println("scratch/images/u.png")
    plt.close()
end

nz = 2^5
H = 1
z = -H:H/(nz - 1):0
z_hr = -H:H/1000:0
bx(z) = exp(-(z + H)/0.01)
ε² = 0.01
Ux = 0
baroclinic_fd_fe(order=2)