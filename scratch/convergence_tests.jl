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
function solve_poisson(V::FESpace, g::Grid, s::StandardElement, f, u₀)
    # create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt
        # calculate contribution to K from element k
        Kᵏ = zeros(s.n_el_nodes, s.n_el_nodes)
        for i=1:s.n_el_nodes
            for j=1:s.n_el_nodes
                Kᵏ[i, j] = dot(s.int_wts, (V.∂φ_int_pts[k, j, 1, :].*V.∂φ_int_pts[k, i, 1, :] .+ V.∂φ_int_pts[k, j, 2, :].*V.∂φ_int_pts[k, i, 2, :]).*V.J_int_pts[k, :])
            end
        end

        # calculate contribution to b from element k
        bᵏ = zeros(s.n_el_nodes)
        for i=1:s.n_el_nodes
            for j=1:s.n_el_nodes
                bᵏ[i] += dot(s.int_wts, f[g.t[k, j]]*s.φ_int_pts[j, :].*s.φ_int_pts[i, :].*V.J_int_pts[k, :])
            end
        end

        # add to global system
        for i=1:s.n_el_nodes
            if g.t[k, i] in g.e
                # edge node, leave for dirichlet
                continue
            end
            for j=1:s.n_el_nodes
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

    # create finite element space for u
    V = FESpace("../meshes/$geo/mesh$nref.h5", order, 4)
    g = V.grid
    x = g.p[:, 1]
    y = g.p[:, 2]

    # mesh resolution 
    h = 1/sqrt(g.np)

    # solution
    ua = @. -exp(x^2)*y

    # pick f such that -∇u = f
    f = @. exp(x^2)*(2 + 4*x^2)*y

    # solve poisson problem
    u = solve_poisson(V, V.grid, V.std_el, f, ua[g.e])

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
    # err = L2norm(V, u - ua)
    err = maximum(abs.(u - ua))
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
    l2 = L2norm(V, u)
"""
function L2norm(V::FESpace, u)
    l2 = 0
    g = V.grid
    s = V.std_el
    for k=1:g.nt
        for i=1:s.n_el_nodes
            for j=1:s.n_el_nodes
                l2 += dot(s.int_wts, u[g.t[k, i]].*s.φ_int_pts[i, :].*u[g.t[k, j]].*s.φ_int_pts[j, :].*V.J_int_pts[k, :])
            end
        end
    end
    return sqrt(l2)
end

# poisson_res(3, 2; plot=true)
# poisson_convergence(0:5)

println("Done.")