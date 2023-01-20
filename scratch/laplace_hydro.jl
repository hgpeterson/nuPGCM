using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
    u = solve_laplace_hydro(g, s, J, δ, f, u₀)

Solve linear system representing the problem
    -δ² u_zz + u = f  on Ω
               u = u₀ on ∂Ω
in a finite element basis.
"""
function solve_laplace_hydro(g, s, J, δ, f, u₀)
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt        
        # calculate contribution to K from element k
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        Kᵏ = δ^2*J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]

        # calculate contribution to M from element k
        Mᵏ = J.dets[k]*s.M

        # calculate contribution to b from element k
        bᵏ = Mᵏ*f[g.t[k, :]]

        # add to global system
        for i=1:g.nn
            for j=1:g.nn
                push!(A, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
            end
            b[g.t[k, i]] += bᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), g.np, g.np)

    # dirichlet along edges
    A, b = add_dirichlet(A, b, g.e, u₀)

    return A\b
end

function laplace_hydro_res(; nref, order, dim, plot=false)
    # parameter
    δ = 0.2

    # get grid
    if dim == 2
        gfile = "../meshes/gmsh/mesh$nref.h5"
    elseif dim == 3
        gfile = "../meshes/bowl3D/mesh$nref.h5"
    end
    g = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)

    # get Jacobians
    J = Jacobians(g1)

    # mesh resolution 
    h = 1/g.np^(1/dim)

    # analytical solution
    if dim == 2
        x = g.p[:, 1]
        y = g.p[:, 2]
        ua = @. 100*sin(x)*(y + 0.25)^2
        f = @. -200*δ^2*sin(x) + ua
    elseif dim == 3
        x = g.p[:, 1]
        y = g.p[:, 2]
        z = g.p[:, 3]
        ua = @. 100*sin(x)*cos(y)*(z + 0.25)^2
        f = @. -200*δ^2*sin(x)*cos(y) + ua
    end

    # dirichlet boundary
    u₀ = ua[g.e]

    # solve laplace_hydro problem
    u = solve_laplace_hydro(g, s, J, δ, f, u₀)

    u = FEField(order, u, g, g1)
    ua = FEField(order, ua, g, g1)

    if plot
        if dim == 2
            quickplot(u, L"u", "images/u.png")
        elseif dim == 3
            write_vtk(g1, "../output/laplace_hydro", ["u"=>u, "error"=>abs(u - ua)])
        end
    end

    # error
    err = L2norm(u - ua, s, J)
    return h, err
end

function laplace_hydro_conv(; nrefs, dim)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u_a||_{L^2}$")
    for o=1:2
        println("Order ", o)
        h = zeros(size(nrefs, 1))
        err = zeros(size(nrefs, 1))
        for i in eachindex(nrefs)
            println("\tRefinement ", nrefs[i])
            h[i], err[i] = laplace_hydro_res(nref=nrefs[i], order=o, dim=dim)
        end
        ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^(o+1)], "k-", alpha=o/3, label=latexstring(L"$h^", o+1, L"$"))
        ax.loglog(h, err, "o", label="Order $o")
    end
    ax.legend(ncol=2, loc=(0.0, 1.05))
    savefig("images/laplace_hydro.png")
    println("images/laplace_hydro.png")
    plt.close()
end

# laplace_hydro_res(nref=2, order=1, dim=3, plot=true)
laplace_hydro_conv(nrefs=0:2, dim=3)