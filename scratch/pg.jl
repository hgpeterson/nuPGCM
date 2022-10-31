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
where j = i-1, k = i-2, and Pₙ is the space of continuous polynomials of degree n.
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
        # ∂z(ux)∂z(vx)
        Kᵏ = abs(J.J[k])*(s.uu.φξφξ*J.ξy[k]^2 + s.uu.φξφη*J.ξy[k]*J.ηy[k] + s.uu.φηφξ*J.ηy[k]*J.ξy[k] + s.uu.φηφη*J.ηy[k]^2)
        # Kwwᵏ = abs(J.J[k])*(s.ww.φξφξ*J.ξy[k]^2 + s.ww.φξφη*J.ξy[k]*J.ηy[k] + s.ww.φηφξ*J.ηy[k]*J.ξy[k] + s.ww.φηφη*J.ηy[k]^2)

        # p*∂x(vx) 
        Cx_puᵏ = abs(J.J[k])*(s.pu.φφξ*J.ξx[k] + s.pu.φφη*J.ηx[k])
        # p*∂z(vz)
        Cz_pwᵏ = abs(J.J[k])*(s.pw.φφξ*J.ξy[k] + s.pw.φφη*J.ηy[k])
        # q*∂x(ux) 
        Cx_upᵏ = abs(J.J[k])*(s.up.φξφ*J.ξx[k] + s.up.φηφ*J.ηx[k])
        # q*∂z(uz)
        Cz_wpᵏ = abs(J.J[k])*(s.wp.φξφ*J.ξy[k] + s.wp.φηφ*J.ηy[k])

        # uy*vx
        Muuᵏ = abs(J.J[k])*s.uu.φφ
        # δ*q*p
        Mppᵏ = abs(J.J[k])*s.pp.φφ

        # b*vz
        rᵏ = abs(J.J[k])*s.ww.φφ*b.values[b.g.t[k, :]]

        # x-mom: ε²*∂z(ux)∂z(vx)
        for i=1:ux.g.nn, j=1:ux.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], uxmap[ux.g.t[k, j]], ε²*Kᵏ[i, j]))
        end
        # y-mom: ∂ε²*z(uy)∂z(vy)
        for i=1:uy.g.nn, j=1:uy.g.nn
            push!(A, (uymap[uy.g.t[k, i]], uymap[uy.g.t[k, j]], ε²*Kᵏ[i, j]))
        end
        # x-mom: -uy*vx
        for i=1:ux.g.nn, j=1:uy.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], uymap[uy.g.t[k, j]], -Muuᵏ[i, j]))
        end
        # y-mom: ux*vy
        for i=1:uy.g.nn, j=1:ux.g.nn
            push!(A, (uymap[uy.g.t[k, i]], uxmap[ux.g.t[k, j]], Muuᵏ[i, j]))
        end
        # x-mom: -p*∂x(vx)
        for i=1:ux.g.nn, j=1:p.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], pmap[p.g.t[k, j]], -Cx_puᵏ[i, j]))
        end
        # cont: ∂x(ux)*q
        for i=1:p.g.nn, j=1:ux.g.nn
            push!(A, (pmap[p.g.t[k, i]], uxmap[ux.g.t[k, j]], Cx_upᵏ[i, j]))
        end
        # z-mom: -p*∂z(vz)
        for i=1:uz.g.nn, j=1:p.g.nn
            push!(A, (uzmap[uz.g.t[k, i]], pmap[p.g.t[k, j]], -Cz_pwᵏ[i, j]))
        end
        # cont: ∂z(uz)*q
        for i=1:p.g.nn, j=1:uz.g.nn
            push!(A, (pmap[p.g.t[k, i]], uzmap[uz.g.t[k, j]], Cz_wpᵏ[i, j]))
        end
        # # uz*vz
        # for i=1:uz.g.nn, j=1:uz.g.nn
        #     push!(A, (uzmap[uz.g.t[k, i]], uzmap[uz.g.t[k, j]], 1e0*ε²*Kwwᵏ[i, j]))
        # end
        # # p*p
        # for i=1:p.g.nn, j=1:p.g.nn
        #     # pressure condition: δ*q*p
        #     push!(A, (pmap[p.g.t[k, i]], pmap[p.g.t[k, j]], 1e-7*Mppᵏ[i, j]))
        # end
        # b
        r[uzmap[uz.g.t[k, :]]] .+= rᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet conditions on bottom and top (replace mom eqtns)
    A, r = add_dirichlet(A, r, uxmap[e.botu], zeros(size(e.botu)))
    A, r = add_dirichlet(A, r, uymap[e.botu], zeros(size(e.botu)))
    A, r = add_dirichlet(A, r, uzmap[e.botw], zeros(size(e.botw)))
    A, r = add_dirichlet(A, r, uzmap[e.topw], zeros(size(e.topw)))

    # pressure condition
    A, r = apply_constraint(A, r, pmap[1], pmap[1], 0)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    # if N < 1000
    #     fig, ax = subplots(1)
    #     ax.imshow(abs.(Matrix(A)) .== 0, cmap="binary_r")
    #     ax.spines["left"].set_visible(false)
    #     ax.spines["bottom"].set_visible(false)
    #     savefig("images/A.png")
    #     println("images/A.png")
    #     plt.close()
    #     # println("Condition number: ", cond(Array(A)))
    # end

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    println(@sprintf("%.1f s", time() - t₀))
    # sol = minres(A, r)
    # sol, ch = bicgstabl(A, r, log=true, verbose=true)
    # sol, ch = lsmr(A, r, log=true, verbose=true)

    # reshape to get u and p
    ux.values[:] = sol[uxmap]
    uy.values[:] = sol[uymap]
    uz.values[:] = sol[uzmap]
    p.values[:] = sol[pmap]
    return ux, uy, uz, p
end

"""
    h, err = pg_res(geo, nref)
"""
function pg_res(geo, nref; plot=false)
    # order of polynomials
    order = 2

    # Ekman number
    # ε² = 1e-5
    # ε² = 1e-4
    ε² = 1e-3
    # ε² = 1e-2
    # ε² = 1e-1
    # ε² = 1

    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"
    gp = FEGrid(gfile, order-2)
    gw = FEGrid(gfile, order-1)
    gu = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    ww = ShapeFunctionIntegrals(gw.s, gw.s)
    pu = ShapeFunctionIntegrals(gp.s, gu.s)
    pw = ShapeFunctionIntegrals(gp.s, gw.s)
    up = ShapeFunctionIntegrals(gu.s, gp.s)
    wp = ShapeFunctionIntegrals(gw.s, gp.s)
    pp = ShapeFunctionIntegrals(gp.s, gp.s)
    s = (uu = uu,
         ww = ww, 
         pu = pu,  
         pw = pw,  
         up = up,  
         wp = wp,
         pp = pp)  
 
    # get Jacobians
    J = Jacobians(g1)   

    println("q⁻¹ = ", sqrt(2*ε²))
    println("h = ", 1/sqrt(g1.np))

    # top and bottom edges
    ebotw, etopw = get_sides(gw)
    ebotu, etopu = get_sides(gu)
    e = (botw = ebotw, topw = etopw, 
         botu = ebotu, topu = etopu)

    # forcing
    function H(x)
        if geo == "gmsh_tri"
            return 1 - abs(x)
        else
            return sqrt(2 - x^2) - 1
        end
    end
    x = gw.p[:, 1] 
    z = gw.p[:, 2] 
    δ = 0.1
    # b = @. z + δ*exp(-(z + H)/δ)
    # b = z
    b = @. δ*exp(-(z + H(x))/δ)

    # initialize FE fields
    ux = FEField(order,   zeros(gu.np), gu, g1)
    uy = FEField(order,   zeros(gu.np), gu, g1)
    uz = FEField(order-1, zeros(gw.np), gw, g1)
    p  = FEField(order-2, zeros(gp.np), gp, g1)
    b  = FEField(order-1, b, gw, g1)

    # solve 
    ux, uy, uz, p = solve_pg(ux, uy, uz, p, b, J, s, e, ε²)

    if plot
        quickplot(ux, L"u^x", "images/ux.png")
        quickplot(uy, L"u^y", "images/uy.png")
        quickplot(uz, L"u^z", "images/uz.png")
        quickplot(p, L"p", "images/p.png")
        plot_profile(ux, 0.5, -H(0.5):1e-3:0, L"u^x", L"z", "images/ux_profile.png")
        plot_profile(uy, 0.5, -H(0.5):1e-3:0, L"u^y", L"z", "images/uy_profile.png")
        plot_profile(uz, 0.5, -H(0.5):1e-3:0, L"u^z", L"z", "images/uz_profile.png")
        plot_profile(p,  0.5, -H(0.5):1e-3:0, L"p",   L"z", "images/p_profile.png")
    end

    return ux, uy, uz, p
end

# ux, uy, uz, p = pg_res("gmsh", 5; plot=true)
# ux, uy, uz, p = pg_res("jc", 5; plot=true)
# ux, uy, uz, p = pg_res("valign", 0; plot=true)
# ux, uy, uz, p = pg_res("", 0; plot=true)

# println(@sprintf("%1.0e  %1.0e  %1.0e", maximum(abs.(ux)), maximum(abs.(uy)), maximum(abs.(uz))))

# x = -1:0.01:1
# H(x) = sqrt(2 - x^2) - 1
# δ = 1e-1
# plot(x, map(x->δ*x/sqrt(2 - x^2)*(1 - exp(-H(x)/δ)), x), label="Thermal Wind")
# plot(x, map(x->fem_evaluate(uy, [x, 0]), x), label="Numerical")
# xlabel(L"x")
# ylabel(L"u^y(0)")
# legend()
# savefig("images/uy0.png")
# println("images/uy0.png")

H(x) = sqrt(2 - x^2) - 1
x = 0.5
z = -H(x):1e-3:0
δ = 1e-1
fig, ax = subplots(1, figsize=(2, 3.2))
ax.plot(map(z->δ*x/sqrt(2 - x^2)*(1 - exp(-(z + H(x))/δ)), z), z)
ax.set_xlabel(L"u^y")
ax.set_ylabel(L"z")
savefig("images/uy_profile_tw.png")
println("images/uy_profile_tw.png")

println("Done.")