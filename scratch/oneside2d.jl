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
  -u_zz = f
   u = u_z = 0 at z = -H
"""
function solve_oneside2d(u, s, J, e, f)
    N = u.g.np
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)
    for k=1:u.g.nt        
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # rhs
        b[u.g.t[k, :]] = M*f.values[u.g.t[k, :]]

        # stamp
        for i=1:u.g.nn, j=1:u.g.nn
            push!(A, (u.g.t[k, i], u.g.t[k, j], K[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # set bottom b.c. u = 0 at z = 0 nodes
    A, b = add_dirichlet(A, b, e.top, e.bot, 0)

    # solve
    u.values[:] = A\b

    return u
end

function oneside2d_res(; nref, order, showplots=false)
    # get grid
    gfile = "../meshes/jc_valign/mesh$nref.h5"
    g = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # edges
    ebot, etop = get_sides(g)
    e = (bot=ebot, top=etop)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)

    # get Jacobians
    J = Jacobians(g1)

    # mesh resolution 
    h = 1/sqrt(g.np)

    # analytical solution
    H(x) = 1 - x^2
    x = g.p[:, 1]
    z = g.p[:, 2]
    f = ones(g.np)
    f = FEField(order, f, g, g1)
    ua = @. -1/2*(z^2 + 2*H(x)*z + H(x)^2)
    ua = FEField(order, ua, g, g1)

    # initialize
    u = FEField(order, zeros(g.np), g, g1)

    # solve 
    u = solve_oneside2d(u, s, J, e, f)

    if showplots
      error = abs(u - ua)
      quickplot(u, L"u", "images/u.png")
      quickplot(ua, L"u_a", "images/ua.png")
      quickplot(error, "Error", "images/error.png")

      plot(x[etop], u.values[etop], "o", ms=1)
      xlabel(L"x")
      ylabel(L"u(z = 0)")
      savefig("images/u_top.png")
      println("images/u_top.png")
      plt.close()

      plot(x[etop], error.values[etop], "o", ms=1)
      xlabel(L"x")
      ylabel(L"Error at $z = 0$")
      savefig("images/error_top.png")
      println("images/error_top.png")
      plt.close()
    end

    # error
    err = L2norm(u - ua, s, J)
    return h, err
end

oneside2d_res(nref=4, order=1, showplots=true)