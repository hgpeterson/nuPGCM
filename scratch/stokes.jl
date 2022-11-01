using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

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
        Kᵏ = abs(J.J[k])*(s.uu.φξφξ*(J.ξx[k]^2       + J.ξy[k]^2) + 
                          s.uu.φξφη*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                          s.uu.φηφξ*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                          s.uu.φηφη*(J.ηx[k]^2       + J.ηy[k]^2))

        # contribution from p*(∇⋅v) term
        Cxᵏ = abs(J.J[k])*(s.pu.φφξ*J.ξx[k] + s.pu.φφη*J.ηx[k])
        Czᵏ = abs(J.J[k])*(s.pu.φφξ*J.ξy[k] + s.pu.φφη*J.ηy[k])

        # contribution from f⋅v
        bxᵏ = abs(J.J[k])*s.uu.φφ*fx.values[ux.g.t[k, :]]
        bzᵏ = abs(J.J[k])*s.uu.φφ*fz.values[ux.g.t[k, :]]

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
    A, b = apply_constraint(A, b, pmap[1], pmap[1], 0)

    # solve
    sol = A\b

    # reshape to get u and p
    ux.values[:] = sol[uxmap]
    uz.values[:] = sol[uzmap]
    p.values[:] = sol[pmap]
    return ux, uz, p
end

"""
    h, err = stokes_res(nref)
"""
function stokes_res(nref; plot=false)
    # order
    order = 3

    # geometry type
    geo = "circle"

    # get grids
    gu = FEGrid("../meshes/$geo/mesh$nref.h5", order)
    gp = FEGrid("../meshes/$geo/mesh$nref.h5", order-1)
    g1 = FEGrid("../meshes/$geo/mesh$nref.h5", 1)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    pu = ShapeFunctionIntegrals(gp.s, gu.s)
    pp = ShapeFunctionIntegrals(gp.s, gp.s)
    s = (uu = uu,
         pu = pu,
         pp = pp)

    # mesh resolution 
    h = 1/sqrt(g1.np)

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
    err_u₁ = H1norm(gu, s.uu, J, ux.values - uxa)
    err_u₂ = H1norm(gu, s.uu, J, uz.values - uza)
    err_p = L2norm(gp, s.pp, J, p.values - pa)
    err= err_u₁ + err_u₂ + err_p
    return h, err
end

"""
    stokes_convergence(nrefs)
"""
function stokes_convergence(nrefs)
    n = size(nrefs, 1)
    h = zeros(n)
    err = zeros(n)
    for i=1:n
        println(nrefs[i])
        h[i], err[i] = stokes_res(nrefs[i])
    end

    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u^a||_{H^1} + ||p - p^a||_{L^2}$")
    # ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^3], "k-", label=L"$h^3$")
    ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^4], "k-", label=L"$h^4$")
    ax.loglog(h, err, "o", label="Data")
    ax.legend()
    ax.set_xlim(0.5*h[end], 2*h[1])
    ax.set_ylim(0.5*err[end], 2*err[1])
    savefig("images/stokes.png")
    println("images/stokes.png")
    plt.close()

    return h, err
end

# stokes_res(0; plot=true)
h, err = stokes_convergence(0:3)