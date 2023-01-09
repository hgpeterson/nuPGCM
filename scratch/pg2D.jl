using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Printf

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    ux, uy, uz, p = solve_pg(ux, uy, uz, p, b, J, s, e, ε²)

PG Inversion:
    -ε²∂zz(ux) - uy + ∂x(p) = 0, 
    -ε²∂zz(uy) + ux         = 0,
                      ∂z(p) = b,
            ∂x(ux) + ∂z(uz) = 0,
with extra condition
    ∫ p dx dy = 0.
Boundary conditions are 
       ux = uy = uz = 0 at z = -H,
    ∂z(ux) = ∂z(uy) = 0 at z = 0, 
                 uz = 0 at z = 0,
Weak form:
    ∫ [ε²*∂z(ux)∂z(ux) - uy*vx - p*∂x(vx) +
       ε²*∂z(uy)∂z(uy) + ux*vy +
       -p*∂z(vz) +
        q*∂x(ux) + q*∂z(uz)
      ] dx dz
    = ∫ b*vz dx dz,
for all 
    ux, vx ∈ Pᵢ,
    uy, vy ∈ Pᵢ, 
    uz, vz ∈ Pⱼ, 
    p, q ∈ Pₖ
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_pg(ux, uy, uz, p, b, J, s, e, ε²)
    # indices
    uxmap = 1:ux.g.np
    uymap = uxmap[end] .+ (1:uy.g.np)
    uzmap = uymap[end] .+ (1:uz.g.np)
    pmap  = uzmap[end] .+ (1:p.g.np)
    N = pmap[end]
    println("N = $N")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:ux.g1.nt
        # element matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        Kᵏ = J.dets[k]*sum(s.uu.K.*JJ, dims=(1, 2))[1, 1, :, :]
        Cxᵏ = J.dets[k]*sum(s.pu.CT.*J.Js[k, :, 1], dims=1)[1, :, :]
        Czᵏ = J.dets[k]*sum(s.pw.CT.*J.Js[k, :, 2], dims=1)[1, :, :]
        Mᵏ = J.dets[k]*s.uu.M

        # rhs
        r[uzmap[uz.g.t[k, :]]] .+= J.dets[k]*s.bw.M*b.values[b.g.t[k, :]]

        # x-mom: ε²*∂z(ux)∂z(vx)
        for i=1:ux.g.nn, j=1:ux.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], uxmap[ux.g.t[k, j]], ε²*Kᵏ[i, j]))
        end
        # y-mom: ε²*∂z(uy)∂z(vy)
        for i=1:uy.g.nn, j=1:uy.g.nn
            push!(A, (uymap[uy.g.t[k, i]], uymap[uy.g.t[k, j]], ε²*Kᵏ[i, j]))
        end
        # x-mom: -uy*vx
        for i=1:ux.g.nn, j=1:uy.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], uymap[uy.g.t[k, j]], -Mᵏ[i, j]))
        end
        # y-mom: ux*vy
        for i=1:uy.g.nn, j=1:ux.g.nn
            push!(A, (uymap[uy.g.t[k, i]], uxmap[ux.g.t[k, j]], Mᵏ[i, j]))
        end
        # x-mom: -p*∂x(vx)
        for i=1:ux.g.nn, j=1:p.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], pmap[p.g.t[k, j]], -Cxᵏ[i, j]))
        end
        # cont: ∂x(ux)*q
        for i=1:p.g.nn, j=1:ux.g.nn
            push!(A, (pmap[p.g.t[k, i]], uxmap[ux.g.t[k, j]], Cxᵏ[j, i]))
        end
        # z-mom: -p*∂z(vz)
        for i=1:uz.g.nn, j=1:p.g.nn
            push!(A, (uzmap[uz.g.t[k, i]], pmap[p.g.t[k, j]], -Czᵏ[i, j]))
        end
        # cont: ∂z(uz)*q
        for i=1:p.g.nn, j=1:uz.g.nn
            push!(A, (pmap[p.g.t[k, i]], uzmap[uz.g.t[k, j]], Czᵏ[j, i]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet conditions on bottom and top (replace mom eqtns)
    A, r = add_dirichlet(A, r, uxmap[e.botu], 0)
    A, r = add_dirichlet(A, r, uymap[e.botu], 0)
    A, r = add_dirichlet(A, r, uzmap[e.botw], 0)
    A, r = add_dirichlet(A, r, uzmap[e.topw], 0)

    # pressure condition
    A, r = add_dirichlet(A, r, pmap[1], 0)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    println(@sprintf("%.1f s", time() - t₀))

    # reshape to get u and p
    ux.values[:] = sol[uxmap]
    uy.values[:] = sol[uymap]
    uz.values[:] = sol[uzmap]
    p.values[:] = sol[pmap]
    return ux, uy, uz, p
end

function pg_res(geo, nref; showplots=false)
    # Ekman number
    # ε² = 1e-5
    # ε² = 1e-4
    # ε² = 1e-3
    # ε² = 1e-2
    # ε² = 1e-1
    ε² = 1

    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"
    gb = FEGrid(gfile, 1)
    gu = FEGrid(gfile, 2)
    gw = FEGrid(gfile, 1)
    gp = FEGrid(gfile, 0)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    pu = ShapeFunctionIntegrals(gp.s, gu.s)
    pw = ShapeFunctionIntegrals(gp.s, gw.s)
    bw = ShapeFunctionIntegrals(gb.s, gw.s)
    s = (uu = uu, 
         pu = pu,  
         pw = pw,
         bw = bw)  
 
    # get Jacobians
    J = Jacobians(g1)   

    println(@sprintf("q⁻¹ = %1.1e", sqrt(2*ε²)))
    println(@sprintf("h   = %1.1e", 1/sqrt(g1.np)))

    # top and bottom edges
    ebotw, etopw = get_sides(gw)
    ebotu, etopu = get_sides(gu)
    e = (botw = ebotw, topw = etopw, 
         botu = ebotu, topu = etopu)

    # forcing
    H(x) = 1 - x^2
    # H(x) = sqrt(2 - x^2) - 1
    x = gb.p[:, 1] 
    z = gb.p[:, 2] 
    δ = 0.1
    # b = @. z + δ*exp(-(z + H(x))/δ)
    # b = z
    b = @. δ*exp(-(z + H(x))/δ)

    # initialize FE fields
    ux = FEField(zeros(gu.np), gu, g1)
    uy = FEField(zeros(gu.np), gu, g1)
    uz = FEField(zeros(gw.np), gw, g1)
    p  = FEField(zeros(gp.np), gp, g1)
    b  = FEField(b,            gb, g1)

    # solve 
    ux, uy, uz, p = solve_pg(ux, uy, uz, p, b, J, s, e, ε²)

    if showplots
        quickplot(ux, L"u^x", "images/ux.png")
        quickplot(uy, L"u^y", "images/uy.png")
        quickplot(uz, L"u^z", "images/uz.png")
        quickplot(p,  L"p",   "images/p.png")
        plot_profile(ux, 0.5, -H(0.5):1e-3:0, L"$u^x$ at $x = 0.5$", L"z", "images/ux_profile.png")
        plot_profile(uy, 0.5, -H(0.5):1e-3:0, L"$u^y$ at $x = 0.5$", L"z", "images/uy_profile.png")
        plot_profile(uz, 0.5, -H(0.5):1e-3:0, L"$u^z$ at $x = 0.5$", L"z", "images/uz_profile.png")
        plot_profile(p,  0.5, -H(0.5):1e-3:0, L"$p$ at $x = 0.5$",   L"z", "images/p_profile.png")
    end

    return ux, uy, uz, p
end

ux, uy, uz, p = pg_res("valign2D", 3; showplots=true)
# ux, uy, uz, p = pg_res("gmsh", 3; showplots=true)
println("Done.")