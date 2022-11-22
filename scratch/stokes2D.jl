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
    ux, uz, p = solve_stokes(ux, uz, p, J, s, fx, fz, ux₀, uz₀)

Stokes problem:
    -Δu + ∇p = f      on Ω,
         ∇⋅u = 0      on Ω,
           u = u₀     on ∂Ω,
with extra condition
    ∫ p dx = 0.
Here u = (ux, uz) is the velocity vector and p is the pressure.
Weak form:
    ∫ (∇u)⊙(∇v) - p (∇⋅v) + q (∇⋅u) dx = ∫ f⋅v dx,
for all 
    vx, vz ∈ P₂ and q ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes(ux, uz, p, J, s, fx, fz, ux₀, uz₀)
    # indices
    uxmap = 1:ux.g.np
    uzmap = uxmap[end] .+ (1:uz.g.np)
    pmap  = uzmap[end] .+ (1:p.g.np)
    N = pmap[end]
    println("N = $N")

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)
    for k=1:ux.g.nt
        # contribution from (∇u)⊙(∇v) term 
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        Kᵏ = J.dets[k]*sum(s.uu.K.*JJ, dims=(1, 2))[1, 1, :, :]

        # contribution from p*(∇⋅v) term
        Cxᵏ = J.dets[k]*sum(s.pu.CT.*J.Js[k, :, 1], dims=1)[1, :, :]
        Czᵏ = J.dets[k]*sum(s.pu.CT.*J.Js[k, :, 2], dims=1)[1, :, :]

        # contribution from f⋅v
        Mᵏ = J.dets[k]*s.uu.M
        bxᵏ = Mᵏ*fx.values[ux.g.t[k, :]]
        bzᵏ = Mᵏ*fz.values[ux.g.t[k, :]]

        # (∇u)⊙(∇v) term
        for i=1:ux.g.nn, j=1:ux.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], uxmap[ux.g.t[k, j]], Kᵏ[i, j]))
        end
        # (∇u)⊙(∇v) term
        for i=1:uz.g.nn, j=1:uz.g.nn
            push!(A, (uzmap[uz.g.t[k, i]], uzmap[uz.g.t[k, j]], Kᵏ[i, j]))
        end
        # -p*(∇⋅v) term
        for i=1:ux.g.nn, j=1:p.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], pmap[p.g.t[k, j]], -Cxᵏ[i, j]))
        end
        # -p*(∇⋅v) term
        for i=1:uz.g.nn, j=1:p.g.nn
            push!(A, (uzmap[uz.g.t[k, i]], pmap[p.g.t[k, j]], -Czᵏ[i, j]))
        end
        # q*(∇⋅u) term 
        for i=1:p.g.nn, j=1:ux.g.nn
            push!(A, (pmap[p.g.t[k, i]], uxmap[ux.g.t[k, j]], Cxᵏ[j, i]))
        end
        for i=1:p.g.nn, j=1:uz.g.nn
            push!(A, (pmap[p.g.t[k, i]], uzmap[uz.g.t[k, j]], Czᵏ[j, i]))
        end
        b[uxmap[ux.g.t[k, :]]] .+= bxᵏ
        b[uzmap[uz.g.t[k, :]]] .+= bzᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet for u along edges
    A, b = add_dirichlet(A, b, uxmap[ux.g.e], ux₀)
    A, b = add_dirichlet(A, b, uzmap[uz.g.e], uz₀)

    # set p to zero somewhere
    A, b = add_dirichlet(A, b, pmap[1], 0)

    # solve
    sol = A\b

    # reshape to get u and p
    ux.values[:] = sol[uxmap]
    uz.values[:] = sol[uzmap]
    p.values[:] = sol[pmap]
    return ux, uz, p
end

function stokes_res(; nref, order, plot=false)
    # get grids
    gfile = "../meshes/circle/mesh$nref.h5"
    gu = FEGrid(gfile, order)
    gp = FEGrid(gfile, order-1)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    pu = ShapeFunctionIntegrals(gp.s, gu.s)
    pp = ShapeFunctionIntegrals(gp.s, gp.s)
    s = (uu = uu,
         pu = pu,
         pp = pp)
         
    # exact solution
    x = gu.p[:, 1] 
    z = gu.p[:, 2] 
    uxa = @.  π/2*cos(π*x/2)*sin(π*z/2)
    uza = @. -π/2*sin(π*x/2)*cos(π*z/2)
    pa = zeros(gp.np)
    fx = @. π^3/4*cos(π*x/2)*sin(π*z/2)
    fz = @. -π^3/4*sin(π*x/2)*cos(π*z/2)

    # dirichlet
    ux₀ = uxa[gu.e]
    uz₀ = uza[gu.e]

    # get Jacobians
    J = Jacobians(g1)

    # initialize FE fields
    ux  = FEField(gu.order, zeros(gu.np), gu, g1)
    uz  = FEField(gu.order, zeros(gu.np), gu, g1)
    p   = FEField(gp.order, zeros(gp.np), gp, g1)
    fx  = FEField(gu.order, fx,           gu, g1)
    fz  = FEField(gu.order, fz,           gu, g1)

    # solve stokes problem
    ux, uz, p = solve_stokes(ux, uz, p, J, s, fx, fz, ux₀, uz₀)

    if plot
        quickplot(ux, L"u^x", "images/ux.png")
        quickplot(uz, L"u^z", "images/uz.png")
        quickplot(p, L"p", "images/p.png")
    end

    # error
    hux = 1/sqrt(gu.np)
    huz = 1/sqrt(gu.np)
    hp = 1/sqrt(gp.np)
    uxa  = FEField(gu.order, uxa, gu, g1)
    uza  = FEField(gu.order, uza, gu, g1)
    pa   = FEField(gp.order, pa,  gp, g1)
    err_ux = L2norm(ux - uxa, s.uu, J)
    err_uz = L2norm(uz - uza, s.uu, J)
    err_p = L2norm(p - pa, s.pp, J)
    return hux, huz, hp, err_ux, err_uz, err_p
end

function stokes_conv(; nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"$L_2$ Error") 
    for i in eachindex(nrefs)
        println(nrefs[i])
        hux, huz, hp, err_ux, err_uz, err_p = stokes_res(nref=nrefs[i], order=2)
        ax.loglog(hux, err_ux, c="tab:blue", "o")
        ax.loglog(huz, err_uz, c="tab:orange", "o")
        ax.loglog(hp, err_p, c="tab:green", "o")
    end
    ax.loglog([1e-1, 1e-2], [5e-3, 5e-3*(1e-1)^3], "k-")
    legend_elements = [
        Line2D([0], [0], color="w", markerfacecolor="tab:blue", marker="o", label=L"u^x"),
        Line2D([0], [0], color="w", markerfacecolor="tab:orange", marker="o", label=L"u^z"),
        Line2D([0], [0], color="w", markerfacecolor="tab:green", marker="o", label=L"p"),
        Line2D([0], [0], color="k", label=L"O(h^3)")
    ]
    ax.legend(handles=legend_elements)
    savefig("images/stokes.png")
    println("images/stokes.png")
    plt.close()
end

# stokes_res(nref=3, order=2, plot=true)
stokes_conv(nrefs=0:3)