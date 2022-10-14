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
    uˣ, uᶻ, p = solve_stokes_z(g1, g2, s22, s12, s11, J, b, ebot1, ebot2, etop1) 

Stokes_z problem:
    -∂zz(uˣ) + ∂x(p) = fˣ,
               ∂z(p) = fᶻ,
     ∂x(uˣ) + ∂z(uᶻ) = 0, 
with extra condition
    p = p0 at index i0
and Dirichlet boundary conditions on u. 
Weak form:
    ∫ [ ∂z(uˣ)∂z(vˣ) + ∂x(p)vˣ 
      + ∂z(p)vᶻ
      + q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ fˣvˣ + fᶻvᶻ dx dz,
for all 
    vˣ ∈ P₂ and q, vᶻ ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_z(g1, g2, s22, s12, s11, J, fx, fz, ux0, uz0, p0, i0) 
    # indices
    uˣmap = 1:g2.np
    uᶻmap = uˣmap[end] .+ (1:g1.np)
    pmap  = uᶻmap[end] .+ (1:g1.np)
    N = pmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # ∂z(u)∂z(v)
        Kᵏ = abs(J.J[k])*(s22.φξφξ*J.ξy[k]^2 + s22.φξφη*J.ξy[k]*J.ηy[k] + s22.φηφξ*J.ηy[k]*J.ξy[k] + s22.φηφη*J.ηy[k]^2)

        # p*∂x(v) and p*∂z(v)
        Cxᵏ = abs(J.J[k])*(s12.φξφ*J.ξx[k] + s12.φηφ*J.ηx[k])
        Czᵏ = abs(J.J[k])*(s11.φξφ*J.ξy[k] + s11.φηφ*J.ηy[k])

        # fv
        rxᵏ = abs(J.J[k])*s22.φφ*fx[g2.t[k, :]]
        rzᵏ = abs(J.J[k])*s11.φφ*fz[g1.t[k, :]]

        # s2*s2
        for i=1:g2.nn
            for j=1:g2.nn
                # x-mom: ∂z(vˣ)∂z(uˣ)
                push!(A, (uˣmap[g2.t[k, i]], uˣmap[g2.t[k, j]], Kᵏ[i, j]))
            end
            r[uˣmap[g2.t[k, i]]] += rxᵏ[i]
        end
        # s2*s1
        for i=1:g2.nn
            for j=1:g1.nn
                # x-mom: ∂x(p)*vˣ
                push!(A, (uˣmap[g2.t[k, i]], pmap[g1.t[k, j]], Cxᵏ[i, j]))
                # cont: ∂x(uˣ)*q
                push!(A, (pmap[g1.t[k, j]], uˣmap[g2.t[k, i]], Cxᵏ[i, j]))
            end
        end
        # s1*s1
        for i=1:g1.nn
            for j=1:g1.nn
                # z-mom: vᶻ*∂z(p)
                push!(A, (uᶻmap[g1.t[k, i]], pmap[g1.t[k, j]], Czᵏ[i, j]))
                # cont: q*∂z(uᶻ)
                push!(A, (pmap[g1.t[k, i]], uᶻmap[g1.t[k, j]], Czᵏ[i, j]))
            end
            r[uᶻmap[g1.t[k, i]]] += rzᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # Dirichlet boundary conditions
    A[uˣmap[g2.e], :] .= 0
    A[diagind(A)[uˣmap[g2.e]]] .= 1
    r[uˣmap[g2.e]] .= ux0

    A[uᶻmap[g1.e], :] .= 0
    A[diagind(A)[uᶻmap[g1.e]]] .= 1
    r[uᶻmap[g1.e]] .= uz0

    # p constraint (replace one of the uᶻ = 0 conditions)
    i = pmap[g1.e[10]]
    A[i, :] .= 0
    A[i, pmap[i0]] = 1
    r[i] = p0

    # solve
    sol = A\r

    # reshape to get u and p
    return sol[uˣmap], sol[uᶻmap], sol[pmap]
end

"""
    h, err = stokes_z_res(nref, order)
"""
function stokes_z_res(nref, order; plot=false)
    # geometry type
    geo = "jc"

    # get shape functions
    s1 = ShapeFunctions(order)
    s2 = ShapeFunctions(order + 1)

    # get shape function integrals
    s11 = ShapeFunctionIntegrals(s1, s1)
    s12 = ShapeFunctionIntegrals(s1, s2)
    s22 = ShapeFunctionIntegrals(s2, s2)

    # get grids
    g0 = Grid("../meshes/$geo/mesh$nref.h5", 1)
    g1 = Grid("../meshes/$geo/mesh$nref.h5", order)
    g2 = Grid("../meshes/$geo/mesh$nref.h5", order + 1)

    # mesh resolution 
    h = 1/sqrt(g2.np)

    # exact solution
    x1 = g1.p[:, 1] 
    z1 = g1.p[:, 2] 
    x2 = g2.p[:, 1] 
    z2 = g2.p[:, 2] 
    uxa = @.  π/2*cos(π*x2/2)*sin(π*z2/2)
    uza = @. -π/2*sin(π*x1/2)*cos(π*z1/2)
    pa = @. exp(x1)*z1^3
    fx = @. exp(x2)*z2^3 + π^3/4*cos(π*x2/2)*sin(π*z2/2)
    fz = @. 3*exp(x1)*z1^2
    i0 = 1
    p0 = pa[i0]

    # dirichlet
    ux0 = uxa[g2.e]
    uz0 = uza[g1.e]

    # get Jacobians
    J = Jacobians(g0)

    # solve stokes_z problem
    uˣ, uᶻ, p = solve_stokes_z(g1, g2, s22, s12, s11, J, fx, fz, ux0, uz0, p0, i0)

    if plot
        quickplot(g2, uˣ, L"u^x", "images/ux.png")
        quickplot(g1, uᶻ, L"u^z", "images/uz.png")
        quickplot(g1, p, L"p", "images/p.png")
        quickplot(g2, uxa, L"u^x", "images/uxa.png")
        quickplot(g1, uza, L"u^z", "images/uza.png")
        quickplot(g1, pa, L"p", "images/pa.png")
    end

    # error
    err_ux = H1norm(g2, s22, J, uˣ - uxa)
    err_uz = H1norm(g1, s11, J, uᶻ - uza)
    err_p = L2norm(g1, s11, J, p - pa)
    err= err_ux + err_uz + err_p
    return h, err
end

h, err = stokes_z_res(3, 1; plot=true)

println("Done.")