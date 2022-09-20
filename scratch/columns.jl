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
    n = size(g.t, 2)
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt        
        # calculate contribution to K from element k
        Kᵏ = δ^2*abs(J.J[k])*(s.φξφξ.*J.ξy[k]^2 + 
                              s.φξφη.*J.ξy[k]*J.ηy[k] +
                              s.φηφξ.*J.ηy[k]*J.ξy[k] +
                              s.φηφη.*J.ηy[k]^2)
        # Kᵏ = δ^2*abs(J.J[k])*(s.φξφξ.*(J.ξx[k]^2       + J.ξy[k]^2) + 
        #                       s.φξφη.*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
        #                       s.φηφξ.*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
        #                       s.φηφη.*(J.ηx[k]^2       + J.ηy[k]^2))

        # calculate contribution to M from element k
        Mᵏ = abs(J.J[k])*s.φφ

        # calculate contribution to b from element k
        bᵏ = sum(Mᵏ, dims=2)

        # add to global system
        for i=1:n
            for j=1:n
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

function convergence(nrefs, order; plots=false)
    # params
    δ = 0.1
    mesh_type = "jc"
    # mesh_type = "gmsh"
    # mesh_type = "circle"

    # save errors
    N = size(nrefs, 1)
    hs = zeros(N)
    errors = zeros(N)
    for k=1:N
        nref = nrefs[k]
        println("refinement ", nref)

        # grid
        g = Grid("../meshes/$mesh_type/mesh$nref.h5", order)
        hs[k] = 1/sqrt(g.np)

        # shape function integrals
        s = ShapeFunctionIntegrals(order)

        # Jacobians
        J = Jacobians(g)

        # solve
        u = solve_columns(g, s, J, δ)
        if plots
            fig, ax, im = tplot(g.p, g.t, u)
            cb = colorbar(im, ax=ax, label=L"u")
            ax.set_xlabel(L"x")
            ax.set_ylabel(L"z")
            savefig("images/u.png")
            println("images/u.png")
            plt.close()
        end

        # compute error
        if mesh_type == "jc"
            H = (sqrt.(2 .- g.p[:, 1].^2) .- 1)
        else
            H = (1 .- g.p[:, 1].^2)
        end
        abs_err = abs.(u - u_exact.(g.p[:, 2], δ, H))

        errors[k] = maximum(abs_err)
    end

    if size(nrefs, 1) > 1
        fig, ax = subplots(1)
        ax.set_xlabel(L"h")
        ax.set_ylabel(L"max $|u - u_a|$")
        ax.plot([hs[1], hs[end]], [errors[1], errors[1]*(hs[end]/hs[1])^2], "k-", label=L"$h^2$")
        ax.plot([hs[1], hs[end]], [errors[1], errors[1]*(hs[end]/hs[1])^3], "k--", label=L"$h^3$")
        ax.loglog(hs, errors, "o", label="Data")
        ax.set_ylim(0.9*errors[end], 1.1*errors[1])
        ax.legend()
        savefig("images/colmuns.png")
        println("images/colmuns.png")
        plt.close()
    end

    return errors
end

function u_exact(z, δ, H)
    return (exp(-z/δ) - exp(H/δ))*(exp(z/δ) - 1)/(1 + exp(H/δ))
end

# errors = convergence(2; plots=true)
errors = convergence(0:4, 2)