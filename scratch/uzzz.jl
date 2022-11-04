using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

Line2D = pyimport("matplotlib.lines").Line2D
plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    u = solve_uzzz()

Solve -∂zzz(u) = f with Dirichlet boundary conditions on u.
Weak form: ∫ ∂zz(u)*∂z(v) dx dz = ∫ f*v dx dz
"""
function solve_uzzz(u, f, J, s, e, u₀)
    # indices
    umap = 1:u.g.np
    N = umap[end]
    println("N = $N")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:u.g1.nt
        # ∂zz(u)∂z(v)
        # Kᵏ = abs(J.J[k])*(s.φξξφξ*J.ξy[k]^3 + 2*s.φξηφξ*J.ξy[k]^2*J.ηy[k] + s.φηηφξ*J.ξy[k]*J.ηy[k]^2 + 
        #                   s.φξξφη*J.ξy[k]^2*J.ηy[k] + 2*s.φξηφη*J.ξy[k]*J.ηy[k]^2 + s.φηηφη*J.ηy[k]^3) 
        Kᵏ = abs(J.J[k])*(s.φξφξ*J.ξy[k]^2 + s.φξφη*J.ξy[k]*J.ηy[k] + s.φηφξ*J.ηy[k]*J.ξy[k] + s.φηφη*J.ηy[k]^2)

        # v*f
        rᵏ = abs(J.J[k])*s.φφ*f.values[f.g.t[k, :]]

        for i=1:u.g.nn, j=1:u.g.nn
            push!(A, (umap[u.g.t[k, i]], umap[u.g.t[k, j]], Kᵏ[i, j]))
        end
        r[umap[u.g.t[k, :]]] += rᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet condition on bottom and top
    A, r = add_dirichlet(A, r, umap[e.bot], u₀.bot)
    A, r = add_dirichlet(A, r, umap[e.top], u₀.top)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    if N < 1000
        M = Matrix(A)
        fig, ax = subplots(1)
        ax.imshow(abs.(M) .== 0, cmap="binary_r")
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        savefig("images/A.png")
        println("images/A.png")
        plt.close()
        println("Condition number: ", cond(M))
        println("rank(A) = ", rank(M))
        println("A is sym: ", issymmetric(M))
    end

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    println(@sprintf("%.1f s", time() - t₀))

    # reshape to get u and p
    u.values[:] = sol[umap]
    return u
end

function uzzz_res(nref, geo; showplots=false, exact=false)
    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"

    gu = FEGrid(gfile, 2)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(gu.s, gu.s)

    # get Jacobians
    J = Jacobians(g1)

    # top and bottom edges
    ebot, etop = get_sides(gu)
    e = (bot=ebot, top=etop)

    if exact 
        # # mesh resolution 
        # h = 1/sqrt(gu.np)

        # # exact solution
        # x = gu.p[:, 1] 
        # z = gu.p[:, 2] 
        # ua = @.  cos(π*x/2)*sin(π*z/2)
        # fx = @. zu*cos(xu*zu)*exp(zu) + π^2/4*cos(π*xu/2)*sin(π*zu/2)
        # fz = @. xw*cos(xw*zw)*exp(zw) + sin(xw*zw)*exp(zw)
        # u₀ = (botw=uza[ebotw], topw=uza[etopw],
        #       botu=uxa[ebotu], topu=uxa[etopu])
    else
        # forcing
        function H(x)
            if geo == "gmsh_tri"
                return 1 - abs(x)
            else
                return sqrt(2 - x^2) - 1
            end
        end
        x = gu.p[:, 1] 
        z = gu.p[:, 2] 
        δ = 0.1
        # f = @. δ*exp(-(z + H(x))/δ)
        f = @. ones(gu.np)
        u₀ = (bot=zeros(size(ebot)), top=zeros(size(etop)))
    end

    # initialize FE fields
    u  = FEField(zeros(gu.np), gu, g1)
    f  = FEField(f,            gu, g1)

    # solve stokes_hydro problem
    u = solve_uzzz(u, f, J, s, e, u₀)

    if showplots
        quickplot(u, L"u", "images/u.png")
        plot_profile(u, 0.5, -H(0.5):1e-3:0, L"$u$ at $x = 0.5$", L"z", "images/u_profile.png")
    end

    if exact
        err_u = L2norm(u.g, s.uu, J, u.values - ua)
        return h, err_u
    else
        return u
    end
end

uzzz_res(0, "jc"; showplots=true)

println("Done.")
