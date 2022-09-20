using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SparseArrays
using SuiteSparse
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    u = solve_poisson(V, f, u₀)

Solves -Δu = f with dirichlet b.c. u = u₀.
"""
function solve_poisson(g::Grid, s::ShapeFunctionIntegrals, J::Jacobians, f, u₀)
    # create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    n = size(g.t, 2)
    for k=1:g.nt
        # calculate contribution to K from element k
        Kᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                Kᵏ[i, j] = abs(J.J[k])*(s.φξφξ[i, j]*(J.ξx[k]^2       + J.ξy[k]^2) + 
                                        s.φξφη[i, j]*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                                        s.φηφξ[i, j]*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                                        s.φηφη[i, j]*(J.ηx[k]^2       + J.ηy[k]^2))
            end
        end

        # calculate contribution to b from element k
        bᵏ = zeros(n)
        for i=1:n
            for j=1:n
                bᵏ[i] += f[g.t[k, j]]*s.φφ[i, j]*abs(J.J[k])
            end
        end

        # add to global system
        for i=1:n
            if g.t[k, i] in g.e
                # edge node, leave for dirichlet
                continue
            end
            for j=1:n
                push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
            end
            b[g.t[k, i]] += bᵏ[i]
        end
    end
    # dirichlet along edges
    for i in g.e
        push!(K, (i, i, 1))
    end
    b[g.e] = u₀

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), g.np, g.np)

    # solve
    return K\b
end

"""
    err, h = poisson_res(nref, order)
"""
function poisson_res(nref, order; plot=false)
    # geometry type
    # geo = "square"
    geo = "circle"

    # get grid
    g = Grid("../meshes/$geo/mesh$nref.h5", order)
    x = g.p[:, 1]
    y = g.p[:, 2]

    # get shape function integrals
    s = ShapeFunctionIntegrals(order)

    # get Jacobians
    J = Jacobians(g)

    # mesh resolution 
    h = 1/sqrt(g.np)

    # solution
    ua = @. -exp(x^2)*y

    # pick f such that -∇u = f
    f = @. exp(x^2)*(2 + 4*x^2)*y

    # solve poisson problem
    u = solve_poisson(g, s, J, f, ua[g.e])

    if plot
        fig, ax, im = tplot(g.p, g.t, u)
        cb = colorbar(im, ax=ax, label=L"u")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/u.png")
        println("images/u.png")
        plt.close()

        fig, ax, im = tplot(g.p, g.t, ua)
        cb = colorbar(im, ax=ax, label=L"u_a")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/ua.png")
        println("images/ua.png")
        plt.close()

        fig, ax, im = tplot(g.p, g.t, abs.(u - ua))
        cb = colorbar(im, ax=ax, label=L"|u - u_a|")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/e.png")
        println("images/e.png")
        plt.close()
    end

    # error
    err = L2norm(g, s, J, u - ua)
    # err = maximum(abs.(u - ua))
    return err, h
end

"""
    poisson_convergence(nrefs)
"""
function poisson_convergence(nrefs)
    hs_l = zeros(size(nrefs, 1))
    hs_q = zeros(size(nrefs, 1))
    err_l = zeros(size(nrefs, 1))
    err_q = zeros(size(nrefs, 1))
    for i in eachindex(nrefs)
        println(nrefs[i])
        err_l[i], hs_l[i] = poisson_res(nrefs[i], 1)
        err_q[i], hs_q[i] = poisson_res(nrefs[i], 2)
    end

    fig, ax = subplots()
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u_a||_{L^2}$")
    ax.loglog([hs_l[1], hs_l[end]], [err_l[1], err_l[1]*(hs_l[end]/hs_l[1])^2], "k-",  label=L"$h^2$")
    ax.loglog([hs_q[1], hs_q[end]], [err_q[1], err_q[1]*(hs_q[end]/hs_q[1])^3], "k--", label=L"$h^3$")
    ax.loglog(hs_l, err_l, "o", label="Linear")
    ax.loglog(hs_q, err_q, "o", label="Quadratic")
    ax.legend(ncol=2)
    ax.set_xlim(0.9*hs_q[end], 1.1*hs_l[1])
    ax.set_ylim(0.5*err_q[end], 2*err_l[1])
    savefig("images/poisson.png")
    println("images/poisson.png")
    plt.close()

    println(@sprintf("Linear: %1.1f", log(err_l[end-1]/err_l[end])/log(hs_l[end-1]/hs_l[end])))
    println(@sprintf("Quad:   %1.1f", log(err_q[end-1]/err_q[end])/log(hs_q[end-1]/hs_q[end])))
end

"""
    l2 = L2norm(g, s, J, u)
"""
function L2norm(g::Grid, s::ShapeFunctionIntegrals, J::Jacobians, u)
    l2 = 0
    n = size(g.t, 2)
    for k=1:g.nt
        for i=1:n
            for j=1:n
                l2 += u[g.t[k, j]]*u[g.t[k, i]]*s.φφ[i, j]*abs(J.J[k])
            end
        end
    end
    return sqrt(l2)
end

# poisson_res(3, 2; plot=true)
poisson_convergence(0:5)

println("Done.")