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
    uˣ, uᶻ, p = solve_stokes_hydro(g0, g1, g2, s, J, b, ebot1, ebot2, etop1, etop2) 

stokes_hydro problem:
    -∂zz(uˣ) + ∂x(p) = 0,
               ∂z(p) = b,
     ∂x(uˣ) + ∂z(uᶻ) = 0, 
with extra condition
    ∫ p dx dz = 0.
Boundary conditions are 
       uˣ = uᶻ = 0 at z = -H,
        ∂z(uˣ) = 0 at z = 0, 
            uᶻ = 0 at z = 0,
Weak form:
    ∫ [ ∂z(uˣ)∂z(vˣ) - p∂x(vˣ) 
      - p∂z(vᶻ)
      + q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ bvᶻ dx dz,
for all 
    vˣ ∈ P₂ and q, vᶻ ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_hydro(g0, g1, g2, s, J, b, ebot1, ebot2, etop1, etop2) 
    # indices
    uˣmap = 1:g2.np
    uᶻmap = uˣmap[end] .+ (1:g1.np)
    pmap  = uᶻmap[end] .+ (1:g0.np)
    N = pmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g0.nt
        # ∂z(uˣ)∂z(vˣ)
        Kᵏ = abs(J.J[k])*(s.uˣuˣ.φξφξ*J.ξy[k]^2 + s.uˣuˣ.φξφη*J.ξy[k]*J.ηy[k] + s.uˣuˣ.φηφξ*J.ηy[k]*J.ξy[k] + s.uˣuˣ.φηφη*J.ηy[k]^2)

        # p*∂x(vˣ) 
        Cx_momᵏ = abs(J.J[k])*(s.puˣ.φφξ*J.ξx[k] + s.puˣ.φφη*J.ηx[k])
        # p*∂z(vᶻ)
        Cz_momᵏ = abs(J.J[k])*(s.puᶻ.φφξ*J.ξy[k] + s.puᶻ.φφη*J.ηy[k])
        # q*∂x(uˣ) 
        Cx_contᵏ = abs(J.J[k])*(s.uˣp.φξφ*J.ξx[k] + s.uˣp.φηφ*J.ηx[k])
        # q*∂z(uᶻ)
        Cz_contᵏ = abs(J.J[k])*(s.uᶻp.φξφ*J.ξy[k] + s.uᶻp.φηφ*J.ηy[k])

        # δ*q*p
        Mᵏ = abs(J.J[k])*s.pp.φφ

        # b*vᶻ
        rᵏ = abs(J.J[k])*s.uᶻuᶻ.φφ*b[g1.t[k, :]]

        # uˣ*vˣ
        for i=1:g2.nn, j=1:g2.nn
            # x-mom: ∂z(uˣ)∂z(vˣ)
            push!(A, (uˣmap[g2.t[k, i]], uˣmap[g2.t[k, j]], Kᵏ[i, j]))
        end
        # p*vˣ
        for i=1:g2.nn, j=1:g0.nn
            # x-mom: -p*∂x(vˣ)
            push!(A, (uˣmap[g2.t[k, i]], pmap[g0.t[k, j]], -Cx_momᵏ[i, j]))
        end
        # uˣ*q
        for i=1:g0.nn, j=1:g2.nn
            # cont: ∂x(uˣ)*q
            push!(A, (pmap[g0.t[k, i]], uˣmap[g2.t[k, j]], Cx_contᵏ[i, j]))
        end
        # uᶻ*q
        for i=1:g0.nn, j=1:g1.nn
            # cont: ∂z(uᶻ)*q
            push!(A, (pmap[g0.t[k, i]], uᶻmap[g1.t[k, j]], Cz_contᵏ[i, j]))
        end
        # p*vᶻ
        for i=1:g1.nn, j=1:g0.nn
            # z-mom: -p*∂z(vᶻ)
            push!(A, (uᶻmap[g1.t[k, i]], pmap[g0.t[k, j]], -Cz_momᵏ[i, j]))
        end
        # p*p
        for i=1:g0.nn, j=1:g0.nn
            # pressure condition: δ*q*p
            push!(A, (pmap[g0.t[k, i]], pmap[g0.t[k, j]], 1e-7*Mᵏ[i, j]))
        end
        # b
        for i=1:g1.nn
            r[uᶻmap[g1.t[k, i]]] += rᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # uˣ = uᶻ = 0 at z = -H (replace mom eqtns at bottom bdy)
    A[uˣmap[ebot2], :] .= 0
    A[diagind(A)[uˣmap[ebot2]]] .= 1
    r[uˣmap[ebot2]] .= 0

    A[uᶻmap[ebot1], :] .= 0
    A[diagind(A)[uᶻmap[ebot1]]] .= 1
    r[uᶻmap[ebot1]] .= 0

    # ∂z(uˣ) = 1 at z = 0 → natural
    # r[uˣmap[etop2]] .+= 1

    # uᶻ = 0 at z = 0 (replace mom eqtn at top bdy)
    A[uᶻmap[etop1], :] .= 0
    A[diagind(A)[uᶻmap[etop1]]] .= 1
    r[uᶻmap[etop1]] .= 0

    # println(rank(A))
    # println(N)

    # imshow(abs.(Matrix(A)) .== 0, cmap="binary_r")
    # savefig("images/A.png")
    # println("images/A.png")
    # plt.close()

    # println(cond(Array(A)))

    # solve
    sol = A\r
    # sol = pinv(Array(A))*r

    # reshape to get u and p
    return sol[uˣmap], sol[uᶻmap], sol[pmap]
end

"""
    h, err = stokes_hydro_res(nref)
"""
function stokes_hydro_res(nref, order; plot=false)
    # geometry type
    geo = "jc"
    # geo = "gmsh"

    # get shape functions
    sp = ShapeFunctions(order-2)
    suᶻ = ShapeFunctions(order-1)
    suˣ = ShapeFunctions(order)

    # get shape function integrals
    uˣuˣ = ShapeFunctionIntegrals(suˣ, suˣ)
    uᶻuᶻ = ShapeFunctionIntegrals(suᶻ, suᶻ)
    puˣ = ShapeFunctionIntegrals(sp, suˣ)
    puᶻ = ShapeFunctionIntegrals(sp, suᶻ)
    uˣp = ShapeFunctionIntegrals(suˣ, sp)
    uᶻp = ShapeFunctionIntegrals(suᶻ, sp)
    pp = ShapeFunctionIntegrals(sp, sp)
    s = (uˣuˣ = uˣuˣ,
         uᶻuᶻ = uᶻuᶻ, 
         puˣ  = puˣ,  
         puᶻ  = puᶻ,  
         uˣp  = uˣp,  
         uᶻp  = uᶻp,
         pp   = pp)  

    # get grids
    g0 = Grid("../meshes/$geo/mesh$nref.h5", order - 2)
    g1 = Grid("../meshes/$geo/mesh$nref.h5", order - 1)
    g2 = Grid("../meshes/$geo/mesh$nref.h5", order)

    # top and bottom edges
    ebot1, etop1 = get_sides(g1)
    ebot2, etop2 = get_sides(g2)

    # forcing
    x = g1.p[:, 1] 
    z = g1.p[:, 2] 
    b = zeros(g1.np)
    # b = @. exp(-x^2/0.1^2 - (z + 0.5)^2/0.1^2)
    # b = @. exp(-(x - 0.5)^2/0.1^2 - (z + 0.75)^2/0.1^2)
    # b = @. exp(-x^2/0.1^2 - (z + 0.2)^2/0.1^2)
    # H_func(x) = lerp(g1.p[ebot1, 1], -g1.p[ebot1, 2], x)
    # H_func(x) = 1 - x^2
    # H = H_func.(x)
    # δ = 0.2
    # b = @. z + δ*H*exp(-(z/H + 1)/δ)
    # b[H .== 0] .= 0
    b = z

    # get Jacobians
    J = Jacobians(g1)

    # solve stokes_hydro problem
    uˣ, uᶻ, p = solve_stokes_hydro(g0, g1, g2, s, J, b, ebot1, ebot2, etop1, etop2)

    if plot
        quickplot(g1, b, g2, uˣ, L"u^x", "images/ux.png")
        quickplot(g1, b, g1, uᶻ, L"u^z", "images/uz.png")
        quickplot(g1, b, g1, b, L"b", "images/b.png")
        quickplot(g1, b, g1, p, L"p", "images/p.png")
    end

    return uˣ, uᶻ, p
end

for i=0:5
    uˣ, uᶻ, p = stokes_hydro_res(i, 2)
    println(@sprintf("%1.e %1.e", maximum(abs.(uˣ)), maximum(abs.(uᶻ))))
end
# uˣ, uᶻ, p = stokes_hydro_res(3, 2; plot=true)

println("Done.")