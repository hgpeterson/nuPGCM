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
    for k=1:g.nt
        # calculate contribution to K from element k
        Kᵏ = abs(J.J[k])*(s.φξφξ*(J.ξx[k]^2       + J.ξy[k]^2) + 
                          s.φξφη*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                          s.φηφξ*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                          s.φηφη*(J.ηx[k]^2       + J.ηy[k]^2))

        # calculate contribution to b from element k
        bᵏ = abs(J.J[k])*s.φφ*f[g.t[k, :]]

        # add to global system
        for i=1:g.nn
            for j=1:g.nn
                push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
            end
            b[g.t[k, i]] += bᵏ[i]
        end
    end

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), g.np, g.np)

    # dirichlet along edges
    K[g.e, :] .= 0
    K[diagind(K)[g.e]] .= 1
    b[g.e] = u₀

    # solve
    return K\b
end

"""
    err, h = poisson_res(nref, order)
"""
function poisson_res(nref, order; plot=false)
    # geometry type
    geo = "square"
    # geo = "circle"

    # get grid
    g = Grid("../meshes/$geo/mesh$nref.h5", order)
    x = g.p[:, 1]
    y = g.p[:, 2]

    # get shape functions
    sf = ShapeFunctions(order; zeromean=false)

    # get shape function integrals
    s = ShapeFunctionIntegrals(sf, sf)

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

        n_profiles = 5
        for i=1:n_profiles
            x = (i - 1)/n_profiles
            H = sqrt(1 - x^2)
            ny = 100
            y = range(-H, H, length=ny)
            u1D = [try fem_evaluate(u, [x, y[j]], g, sf) catch NaN end for j=1:ny]
            ua1D = -exp(x^2)*y
            fig, ax = subplots(figsize=(1.955, 3.167))
            ax.plot(u1D, y, label="Numerical")
            ax.plot(ua1D, y, "k--", lw=0.5, label="Analytical")
            ax.set_xlabel(L"u")
            ax.set_ylabel(L"y")
            ax.set_title(latexstring(L"$x = $", @sprintf("%1.1f", x)))
            ax.legend()
            savefig("images/u_profile$i.png")
            println("images/u_profile$i.png")
            plt.close()
        end
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
    # ax.set_ylabel(L"Error $||u - u_a||_{L^\infty}$")
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

# poisson_res(3, 2; plot=true)
poisson_convergence(0:5)

println("Done.")