using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    uˣ, uᶻ, p = solve_stokes_z(g1, g2, s22, s12, s11, J, b, ebot1, ebot2, etop1) 

Stokes_z problem:
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
    ∫ [ ∂z(uˣ)∂z(vˣ) + ∂x(p)vˣ 
      + ∂z(p)vᶻ
      + q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ bvᶻ dx dz,
for all 
    vˣ ∈ P₂ and q, vᶻ ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_z(g1, g2, s22, s12, s11, J, b, ebot1, ebot2, etop1) 
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
        rᵏ = abs(J.J[k])*s11.φφ*b[g1.t[k, :]]

        # s2*s2
        for i=1:g2.nn
            for j=1:g2.nn
                # x-mom: ∂z(vˣ)∂z(uˣ)
                push!(A, (uˣmap[g2.t[k, i]], uˣmap[g2.t[k, j]], Kᵏ[i, j]))
            end
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

    # ∂z(uˣ) = 0 at z = 0 → natural

    # uᶻ = 0 at z = 0 (replace continuity at top bdy)
    A[pmap[etop1], :] .= 0
    for e in etop1
        A[pmap[e], uᶻmap[e]] = 1
    end
    r[pmap[etop1]] .= 0

    # p constraint (replace one of the uᶻ = 0 conditions at top bdy)
    n = convert(Int64, round(size(etop1, 1)/2)) # middle of top bdy
    i = pmap[etop1[n]]
    A[i, :] .= 0
    A[i, i] = 1
    # A[i, pmap[:]] .= 1
    # for k=1:g1.nt
    #     A[i, pmap[g1.t[k, :]]] += sum(s11.φφ, dims=1)'
    # end
    r[i] = 0

    # println(rank(A))
    # println(N)

    # solve
    sol = A\r

    # reshape to get u and p
    return sol[uˣmap], sol[uᶻmap], sol[pmap]
end

"""
    h, err = stokes_z_res(nref)
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

    # top and bottom edges
    etop1 = g1.e[abs.(g1.p[g1.e, 2]) .< 1e-4]
    ebot1 = g1.e[abs.(g1.p[g1.e, 2]) .>= 1e-4]
    eleft1 = g1.e[abs.(g1.p[g1.e, 1] .+ 1) .<= 1e-4]
    eright1 = g1.e[abs.(g1.p[g1.e, 1] .- 1) .<= 1e-4]
    deleteat!(etop1, findall(x->x==eleft1[1], etop1))
    deleteat!(etop1, findall(x->x==eright1[1], etop1))
    push!(ebot1, eleft1[1])
    push!(ebot1, eright1[1])

    ebot2 = g2.e[abs.(g2.p[g2.e, 2]) .>= 1e-4]
    eleft2 = g2.e[abs.(g2.p[g2.e, 1] .+ 1) .<= 1e-4]
    eright2 = g2.e[abs.(g2.p[g2.e, 1] .- 1) .<= 1e-4]
    push!(ebot2, eleft2[1])
    push!(ebot2, eright2[1])

    # mesh resolution 
    h = 1/sqrt(g2.np)

    # forcing
    x = g1.p[:, 1] 
    z = g1.p[:, 2] 
    b = @. exp(-x^2/0.1^2 - (z + 0.2)^2/0.1^2)

    # get Jacobians
    J = Jacobians(g0)

    # solve stokes_z problem
    uˣ, uᶻ, p = solve_stokes_z(g1, g2, s22, s12, s11, J, b, ebot1, ebot2, etop1)

    if plot
        quickplot(g1, b, g2, uˣ, L"u^x", "images/ux.png")
        quickplot(g1, b, g1, uᶻ, L"u^z", "images/uz.png")
        quickplot(g1, b, g1, p, L"p", "images/p.png")
        quickplot(g1, b, g1, b, L"b", "images/b.png")
    end

    # error
    err = NaN
    return uˣ, uᶻ, p
end

"""
    quickplot(g, u, clabel, ofile)
"""
function quickplot(gb, b, gu, u, clabel, ofile)
    fig, ax, im = tplot(gu.p, gu.t, u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.tricontour(gb.p[:, 1], gb.p[:, 2], gb.t[:, 1:3] .- 1, b, linewidths=0.5, colors="k", linestyles="-", alpha=0.3)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end

stokes_z_res(4, 1; plot=true)

println("Done.")