using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
    u = solve_columns(g, s, J, δ)

Solve linear system representing the problem
    -δ² u_zz + u = 1
in a finite element basis.
"""
function solve_columns(g::Grid, s::ShapeFunctionIntegrals, J::Jacobians, δ)
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt        
        # calculate contribution to K from element k
        Kᵏ = δ^2*abs(J.J[k])*(s.φξφξ.*J.ξy[k]^2 + 
                              s.φξφη.*J.ξy[k]*J.ηy[k] +
                              s.φηφξ.*J.ηy[k]*J.ξy[k] +
                              s.φηφη.*J.ηy[k]^2)

        # calculate contribution to M from element k
        Mᵏ = abs(J.J[k])*s.φφ

        # calculate contribution to b from element k
        bᵏ = sum(Mᵏ, dims=2)

        # add to global system
        for i=1:g.nn
            for j=1:g.nn
                if g.t[k, i] in g.e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(A, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
            end
            b[g.t[k, i]] += bᵏ[i]
        end
    end

    # dirichlet u = 0 along edges
    for i in g.e
        push!(A, (i, i, 1))
    end
    b[g.e] .= 0

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), g.np, g.np)

    return A\b
end

"""
    err, h = columns_res(nref, order)
"""
function columns_res(nref, order; plot=false)
    # parameter
    δ = 0.05

    # geometry type
    geo = "jc"

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

    # solve columns problem
    u = solve_columns(g, s, J, δ)

    # analytical solution
    H = sqrt.(2 .- g.p[:, 1].^2) .- 1
    ua = u_exact.(g.p[:, 2], δ, H)

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
            H = sqrt(2 - x^2) - 1
            nz = 100
            z = range(-H, 0, length=nz)
            u1D = [try fem_evaluate(u, [x, z[j]], g) catch NaN end for j=1:nz]
            ua1D = u_exact.(z, δ, H)
            fig, ax = subplots(figsize=(1.955, 3.167))
            ax.plot(u1D, z, label="Numerical")
            ax.plot(ua1D, z, "k--", lw=0.5, label="Analytical")
            ax.set_xlabel(L"u")
            ax.set_ylabel(L"z")
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
    columns_convergence(nrefs)
"""
function columns_convergence(nrefs)
    hs_l = zeros(size(nrefs, 1))
    hs_q = zeros(size(nrefs, 1))
    err_l = zeros(size(nrefs, 1))
    err_q = zeros(size(nrefs, 1))
    for i in eachindex(nrefs)
        println(nrefs[i])
        err_l[i], hs_l[i] = columns_res(nrefs[i], 1)
        err_q[i], hs_q[i] = columns_res(nrefs[i], 2)
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
    savefig("images/columns.png")
    println("images/columns.png")
    plt.close()

    println(@sprintf("Linear: %1.1f", log(err_l[end-1]/err_l[end])/log(hs_l[end-1]/hs_l[end])))
    println(@sprintf("Quad:   %1.1f", log(err_q[end-1]/err_q[end])/log(hs_q[end-1]/hs_q[end])))
end

function u_exact(z, δ, H)
    return (exp(-z/δ) - exp(H/δ))*(exp(z/δ) - 1)/(1 + exp(H/δ))
end

columns_convergence(0:5)
# columns_res(1, 1; plot=true)