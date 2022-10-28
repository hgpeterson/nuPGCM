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
    uˣ, uʸ, uᶻ, p = solve_pg(g, s, J, b, e, ε²)

PG Inversion:
    -ε²∂zz(uˣ) - uʸ + ∂x(p) = 0, 
    -ε²∂zz(uʸ) + uˣ         = 0,
                      ∂z(p) = b,
            ∂x(uˣ) + ∂z(uᶻ) = 0,
with extra condition
    ∫ p dx dy = 0.
Boundary conditions are 
       uˣ = uʸ = uᶻ = 0 at z = -H,
    ∂z(uˣ) = ∂z(uʸ) = 0 at z = 0, 
                 uᶻ = 0 at z = 0,
Weak form:
    ∫ [ε²∂z(uˣ)∂z(uˣ) - uʸvˣ - p∂x(vˣ) +
       ε²∂z(uʸ)∂z(uʸ) + uˣvʸ +
       -p∂z(vᶻ) +
        q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ bvᶻ dx dz,
for all 
    uˣ, vˣ ∈ Pᵢ,
    uʸ, vʸ ∈ Pᵢ, 
    uᶻ, vᶻ ∈ Pⱼ, 
    p, q ∈ Pₖ
where j = i-1, k = i-2, and Pₙ is the space of continuous polynomials of degree n.
"""
function solve_pg(uˣ, uʸ, uᶻ, p, b, J, s, e, ε²)
    # indices
    uˣmap = 1:uˣ.g.np
    uʸmap = uˣmap[end] .+ (1:uʸ.g.np)
    uᶻmap = uʸmap[end] .+ (1:uᶻ.g.np)
    pmap  = uᶻmap[end] .+ (1:p.g.np)
    N = pmap[end]
    println("N = $N")

    # stamp system
    println("Building...")
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:uˣ.g1.nt
        # ∂z(uˣ)∂z(vˣ)
        Kᵏ = abs(J.J[k])*(s.uu.φξφξ*J.ξy[k]^2 + s.uu.φξφη*J.ξy[k]*J.ηy[k] + s.uu.φηφξ*J.ηy[k]*J.ξy[k] + s.uu.φηφη*J.ηy[k]^2)

        # p*∂x(vˣ) 
        Cx_puᵏ = abs(J.J[k])*(s.pu.φφξ*J.ξx[k] + s.pu.φφη*J.ηx[k])
        # p*∂z(vᶻ)
        Cz_pwᵏ = abs(J.J[k])*(s.pw.φφξ*J.ξy[k] + s.pw.φφη*J.ηy[k])
        # q*∂x(uˣ) 
        Cx_upᵏ = abs(J.J[k])*(s.up.φξφ*J.ξx[k] + s.up.φηφ*J.ηx[k])
        # q*∂z(uᶻ)
        Cz_wpᵏ = abs(J.J[k])*(s.wp.φξφ*J.ξy[k] + s.wp.φηφ*J.ηy[k])

        # uʸvˣ
        Muuᵏ = abs(J.J[k])*s.uu.φφ
        # δ*q*p
        Mppᵏ = abs(J.J[k])*s.pp.φφ

        # b*vᶻ
        rᵏ = abs(J.J[k])*s.ww.φφ*b.values[b.g.t[k, :]]

        # u*u
        for i=1:uˣ.g.nn, j=1:uˣ.g.nn
            # x-mom: ∂z(uˣ)∂z(vˣ)
            push!(A, (uˣmap[uˣ.g.t[k, i]], uˣmap[uˣ.g.t[k, j]], ε²*Kᵏ[i, j]))
            # y-mom: ∂z(uʸ)∂z(vʸ)
            push!(A, (uʸmap[uʸ.g.t[k, i]], uʸmap[uʸ.g.t[k, j]], ε²*Kᵏ[i, j]))
            # x-mom: uʸ*vˣ
            push!(A, (uˣmap[uˣ.g.t[k, i]], uʸmap[uʸ.g.t[k, j]], -Muuᵏ[i, j]))
            # y-mom: uˣ*vʸ
            push!(A, (uʸmap[uʸ.g.t[k, i]], uˣmap[uˣ.g.t[k, j]], Muuᵏ[i, j]))
        end
        # p*vˣ
        for i=1:uˣ.g.nn, j=1:p.g.nn
            # x-mom: -p*∂x(vˣ)
            push!(A, (uˣmap[uˣ.g.t[k, i]], pmap[p.g.t[k, j]], -Cx_puᵏ[i, j]))
        end
        # uˣ*q
        for i=1:p.g.nn, j=1:uˣ.g.nn
            # cont: ∂x(uˣ)*q
            push!(A, (pmap[p.g.t[k, i]], uˣmap[uˣ.g.t[k, j]], Cx_upᵏ[i, j]))
        end
        # p*vᶻ
        for i=1:uᶻ.g.nn, j=1:p.g.nn
            # z-mom: -p*∂z(vᶻ)
            push!(A, (uᶻmap[uᶻ.g.t[k, i]], pmap[p.g.t[k, j]], -Cz_pwᵏ[i, j]))
        end
        # uᶻ*q
        for i=1:p.g.nn, j=1:uᶻ.g.nn
            # cont: ∂z(uᶻ)*q
            push!(A, (pmap[p.g.t[k, i]], uᶻmap[uᶻ.g.t[k, j]], Cz_wpᵏ[i, j]))
        end
        # # p*p
        # for i=1:p.g.nn, j=1:p.g.nn
        #     # pressure condition: δ*q*p
        #     push!(A, (pmap[p.g.t[k, i]], pmap[p.g.t[k, j]], 1e-7*Mppᵏ[i, j]))
        # end
        # b
        r[uᶻmap[uᶻ.g.t[k, :]]] .+= rᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet conditions on bottom and top (replace mom eqtns)
    A, r = add_dirichlet(A, r, uˣmap[e.botu], zeros(size(e.botu)))
    A, r = add_dirichlet(A, r, uʸmap[e.botu], zeros(size(e.botu)))
    A, r = add_dirichlet(A, r, uᶻmap[e.botw], zeros(size(e.botw)))
    A, r = add_dirichlet(A, r, uᶻmap[e.topw], zeros(size(e.topw)))

    # pressure condition
    A, r = apply_constraint(A, r, pmap[1], pmap[1], 0)

    # remove zeros
    dropzeros!(A)

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
    println("Solving...")
    sol = A\r
    # sol = minres(A, r)
    # sol, ch = bicgstabl(A, r, log=true, verbose=true)
    # sol, ch = lsmr(A, r, log=true, verbose=true)

    # reshape to get u and p
    uˣ.values[:] = sol[uˣmap]
    uʸ.values[:] = sol[uʸmap]
    uᶻ.values[:] = sol[uᶻmap]
    p.values[:] = sol[pmap]
    return uˣ, uʸ, uᶻ, p
end

"""
    h, err = pg_res(geo, nref)
"""
function pg_res(geo, nref; plot=false)
    # order of polynomials
    order = 2

    # Ekman number
    # ε² = 1e-4
    # ε² = 1e-2
    ε² = 1

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

    println("δ = ", sqrt(2*ε²))
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
    uˣ = FEField(order,   zeros(gu.np), gu, g1)
    uʸ = FEField(order,   zeros(gu.np), gu, g1)
    uᶻ = FEField(order-1, zeros(gw.np), gw, g1)
    p  = FEField(order-2, zeros(gp.np), gp, g1)
    b  = FEField(order-1, b, gw, g1)

    # solve 
    uˣ, uʸ, uᶻ, p = solve_pg(uˣ, uʸ, uᶻ, p, b, J, s, e, ε²)

    if plot
        quickplot(b, uˣ, L"u^x", "images/ux.png")
        quickplot(b, uʸ, L"u^y", "images/uy.png")
        quickplot(b, uᶻ, L"u^z", "images/uz.png")
        quickplot(b, p, L"p", "images/p.png")
        plot_profile(uˣ, 0.5, -H(0.5)+1e-5:1e-3:0, L"u^x", L"z", "images/ux_profile.png")
        plot_profile(uʸ, 0.5, -H(0.5)+1e-5:1e-3:0, L"u^y", L"z", "images/uy_profile.png")
        plot_profile(uᶻ, 0.5, -H(0.5)+1e-5:1e-3:0, L"u^z", L"z", "images/uz_profile.png")
        plot_profile(p, 0.5, -H(0.5)+1e-5:1e-3:0, L"p", L"z", "images/p_profile.png")
    end

    return uˣ, uʸ, uᶻ, p
end

# b = z
# δ     ε²    res  uˣ    uʸ    uᶻ
# 1e-1  1e-4  0    4e-2  4e-1  2e-2
# 1e-1  1e-4  1    3e-1  4e-1  1e-1
# 1e-1  1e-4  2    2e-1  2e-1  1e-1   # not high enough resolution to handle
# 1e-1  1e-4  3    9e-2  6e-2  7e-2
# 1e-1  1e-4  4    3e-2  1e-2  3e-2

# 1e-1  1e-2  0    3e-2  2e-2  1e-2   # uˣ and uʸ ~ O(h^2) but not uᶻ always
# 1e-1  1e-2  1    1e-2  8e-3  9e-3
# 1e-1  1e-2  2    3e-3  2e-3  3e-3
# 1e-1  1e-2  3    1e-3  6e-4  3e-3
# 1e-1  1e-2  4    3e-4  1e-4  3e-4

# 1e-1  1e0   0    3e-4  4e-6  1e-4   # uˣ and uʸ ~ O(h^2) but not uᶻ always
# 1e-1  1e0   1    1e-4  2e-6  1e-4
# 1e-1  1e0   2    3e-5  4e-7  3e-5
# 1e-1  1e0   3    1e-5  8e-8  3e-5
# 1e-1  1e0   4    3e-6  2e-8  3e-6

# b = δ*exp(-(z + H)/δ)
# δ     ε²    uˣ    uʸ    uᶻ
# 1e-1  1e-4  8e-3  3e-2  6e-3
# 1e-1  1e-2  4e-3  6e-3  2e-3
# 1e-1  1e0   5e-5  9e-7  2e-5

# uˣ, uʸ, uᶻ, p = pg_res("gmsh", 0; plot=true)
uˣ, uʸ, uᶻ, p = pg_res("jc", 3; plot=true)
# uˣ, uʸ, uᶻ, p = pg_res("valign", 0; plot=true)
# uˣ, uʸ, uᶻ, p = pg_res("", 0; plot=true)

# println(@sprintf("%1.0e  %1.0e  %1.0e", maximum(abs.(uˣ)), maximum(abs.(uʸ)), maximum(abs.(uᶻ))))

println("Done.")