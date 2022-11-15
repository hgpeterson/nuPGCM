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
    ω, χ = solve_stokes_hydro_vort(ω, χ, f, J, s, e)

Solve
       -∂zz(ω) = f,
   -∂zz(χ) - ω = 0,
with boundary conditions 
    ω = χ = 0  at  z = 0,
    ∂x(χ) = ∂z(χ) = 0  at  z = -H.
"""
function solve_stokes_hydro_vort(ω, χ, f, J, s, e)
    # unpack grids
    g1 = ω.g1
    g = ω.g
    # indices
    ωmap = 1:g.np
    χmap = (g.np+1):2*g.np
    N = 2*g.np
    println("N = $N")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # stiffness and mass matrices
        K = abs(J.J[k])*(s.φξφξ*J.ξy[k]^2 + s.φξφη*J.ξy[k]*J.ηy[k] + s.φηφξ*J.ηy[k]*J.ξy[k] + s.φηφη*J.ηy[k]^2)
        M = abs(J.J[k])*s.φφ

        # f
        r[ωmap[g.t[k, :]]] += M*f.values[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            # -∂zz(ω)
            push!(A, (ωmap[g.t[k, i]], ωmap[g.t[k, j]], K[i, j]))
            # -∂zz(χ)
            push!(A, (χmap[g.t[k, i]], χmap[g.t[k, j]], K[i, j]))
            # -ω
            push!(A, (χmap[g.t[k, i]], ωmap[g.t[k, j]], M[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # top: dirichlet 
    A, r = add_dirichlet(A, r, ωmap[e.top], 0)
    A, r = add_dirichlet(A, r, χmap[e.top], 0)

    # # bot: dirichlet 
    # A, r = add_dirichlet(A, r, ωmap[e.bot], χmap[e.bot], 0)

    # special dirichlet condition ∂x(χ) = 0 at z = -H
    edges, boundary_indices, emap = all_edges(g1.t)
    w_quad, t_quad = quad_weights_points(2*g.order-1, 1)
    ps = standard_element_nodes(g.order)
    A[ωmap[e.bot], :] .= 0
    r[ωmap[e.bot]] .= 0
    for k=1:g1.nt, ie=1:3
        if emap[k, ie] in boundary_indices # edge `ie` of triangle `k` is on the boundary
            # get local indices of each point on edge `ie`:
            if g.order == 1
                il = [ie, mod1(ie+1, 3)]
            elseif g.order == 2
                il = [ie, ie+3, mod1(ie+1, 3)]
            end
            ig = g.t[k, il]
            if (ig[1] in e.bot) && (ig[end] in e.bot) # the edge is on the *bottom* boundary
                # get global coordinates of end points on edge
                p1 = g.p[ig[1], :]
                p2 = g.p[ig[end], :]

                # get local coordinates on standard triangle of each point on edge
                ξ1 = ps[il[1], :]
                ξ2 = ps[il[end], :]

                # compute ∫ φᵢ(ξ(t))*∂x(φⱼ(ξ(t)))*||ξ′(t)||*dt for t ∈ [-1, 1] where ξ(-1) = ξ1 and ξ(1) = ξ2
                ξ(t) = (ξ2 - ξ1)/2*t + (ξ2 + ξ1)/2
                for i=il, j=1:g.nn
                    f1(t) = φ(g.s, i, ξ(t))*(φξ(g.s, j, ξ(t))*J.ξx[k] + φη(g.s, j, ξ(t))*J.ηx[k])*norm(p2 - p1)/2
                    # f1(t) = φ(g.s, i, ξ(t))*φξ(g.s, j, ξ(t))*norm(p2 - p1)/(p2[1] - p1[1])/2
                    ∫f = dot(w_quad, f1.(t_quad))
                    A[ωmap[g.t[k, i]], χmap[g.t[k, j]]] += ∫f
                end
            end
        end
    end

    # corners: dirichlet 
    A, r = add_dirichlet(A, r, ωmap[e.bot[1]], 0)
    A, r = add_dirichlet(A, r, ωmap[e.bot[end]], 0)
    A, r = add_dirichlet(A, r, χmap[e.bot[1]], 0)
    A, r = add_dirichlet(A, r, χmap[e.bot[end]], 0)

    # off-corners: dirichlet
    # FIXME: need a better way of finding nodes on off-corners
    println(g.p[e.bot[3], :])
    println(g.p[e.bot[end-1], :])
    A, r = add_dirichlet(A, r, ωmap[e.bot[3]], χmap[e.bot[3]], 0)
    A, r = add_dirichlet(A, r, ωmap[e.bot[end-1]], χmap[e.bot[end-1]], 0)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    R = rank(A)
    println("rank(A): ", R, " = N - ", N - R)
    if R < N
        if N > 2000
            error("🐻")
        end
        null = nullspace(Matrix(A))
        ω.values[:] = null[ωmap]
        χ.values[:] = null[χmap]
        return ω, χ
    end

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    ω.values[:] = sol[ωmap]
    χ.values[:] = sol[χmap]
    println(@sprintf("%.1f s", time() - t₀))

    return ω, χ
end

function stokes_hydro_vort_res(geo, nref; showplots=false)
    # order of polynomials
    order = 1
    # order = 2

    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"
    g  = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    # top and bottom edges
    ebot, etop = get_sides(g)
    e = (bot = ebot, top = etop) 

    # forcing
    x = g.p[:, 1] 
    z = g.p[:, 2] 
    H(x) = sqrt(2 - x^2) - 1
    Hx(x) = -x/sqrt(2 - x^2)
    δ = 1
    f = @. -Hx(x)*exp(-(z + H(x))/δ)
    # f = -ones(g.np)

    # initialize FE fields
    ω = FEField(zeros(g.np), g, g1)
    χ = FEField(zeros(g.np), g, g1)
    f  = FEField(f,          g, g1)

    # solve 
    ω, χ = solve_stokes_hydro_vort(ω, χ, f, J, s, e)

    if showplots
        quickplot(ω, L"\omega", "images/omega.png")
        quickplot(χ, L"\chi", "images/chi.png")
    end

    ω_a = @. -1/8*z*(-3 + 3*sqrt(2 - x^2) + 4*z)
    χ_a = @. -1/48*(-1 + sqrt(2 - x^2) - 2*z)*z*(-1 + sqrt(2 - x^2) + z)^2
    ω_a = FEField(ω_a, g, g1)
    χ_a = FEField(χ_a, g, g1)
    quickplot(ω_a, L"\omega_a", "images/omega_a.png")
    quickplot(χ_a, L"\chi_a", "images/chi_a.png")
    println(@sprintf("ω error: %.1e", maximum(abs.(ω.values - ω_a.values))))
    println(@sprintf("χ error: %.1e", maximum(abs.(χ.values - χ_a.values))))

    return ω, χ
end

ω, χ = stokes_hydro_vort_res("gmsh", 5; showplots=true)
# ω, χ = stokes_hydro_vort_res("jc", 5; showplots=true)

println("Done.")