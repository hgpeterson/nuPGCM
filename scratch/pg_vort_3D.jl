using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Printf
using HDF5

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, b, J, s, e, ε²)

PG Inversion:
    -ε²∂zz(ωx) - f*ωy =  ∂y(b), 
    -ε²∂zz(ωy) + f*ωx = -∂x(b),
       ∂zz(χx) + ωx = 0,
       ∂zz(χy) + ωy = 0,
with boundary conditions 
       ωx = -τy, ωy = -τx  at  z = 0,
              χx = χy = 0  at  z = 0,
      ∂z(χx) = ∂z(χy) = 0  at  z = -H,
      ∂x(χy) - ∂y(χx) = 0  at  z = -H,
      -ε²*(∂x(τy) - ∂y(τx)) - ε²*(∂x(ωx) - ∂y(ωy)) - β*χx = 0  at  z = -H.
For now, we simplify the problem so that
    - f = 1,
    - τx = τy = 0, and
    - b.c.'s 4 and 5 are just χy = ωx = 0 at z = -H.
"""
function solve_pg_vort(ωx, ωy, χx, χy, b, J, s, bdy, ε²)
    # unpack grids
    g1 = ωx.g1
    g = ωx.g
    # indices
    ωxmap = 1:g.np
    ωymap = (g.np+1):2*g.np
    χxmap = (2*g.np+1):3*g.np
    χymap = (3*g.np+1):4*g.np
    N = 4*g.np
    println("N = $N")

    # # tag whether node of triangle is on boundary or not
    # edge_tags = zeros(Bool, size(g.t))
    # for k=1:g.nt
    #     for i in axes(g.t, 2)
    #         edge_tags[k, i] = g.t[k, i] in g.e
    #     end
    # end

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # stiffness matrix
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]

        # mass matrix
        M = J.dets[k]*s.M

        # ∂y(b) and -∂x(b)
        Cx = J.dets[k]*sum(s.C.*J.Js[k, :, 1], dims=1)[1, :, :]
        Cy = J.dets[k]*sum(s.C.*J.Js[k, :, 2], dims=1)[1, :, :]
        r[ωxmap[g.t[k, :]]] += Cy*b.values[g.t[k, :]]
        r[ωymap[g.t[k, :]]] -= Cx*b.values[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            # if edge_tags[k, i]
            #     # edge node, leave for dirichlet
            #     continue
            # end

            # indices
            ωxi = ωxmap[g.t[k, :]]
            ωyi = ωymap[g.t[k, :]]
            χxi = χxmap[g.t[k, :]]
            χyi = χymap[g.t[k, :]]

            # -ε²*∂zz(ωx)
            push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
            # -ωy
            push!(A, (ωxi[i], ωyi[j], -M[i, j]))

            # -ε²*∂zz(ωy)
            push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
            # +ωx
            push!(A, (ωyi[i], ωxi[j], M[i, j]))

            # +∂zz(χx)
            push!(A, (χxi[i], χxi[j], -K[i, j]))
            # +ωx
            push!(A, (χxi[i], ωxi[j], M[i, j]))

            # +∂zz(χy)
            push!(A, (χyi[i], χyi[j], -K[i, j]))
            # +ωy
            push!(A, (χyi[i], ωyi[j], M[i, j]))
        end
    end

    # # top b.c.
    # for i in eachindex(e.top)
    #     ie = e.top[i]
    #     push!(A, (ωxmap[ie], ωxmap[ie], 1))
    #     push!(A, (ωymap[ie], ωymap[ie], 1))
    #     push!(A, (χxmap[ie], χxmap[ie], 1))
    #     push!(A, (χymap[ie], χymap[ie], 1))
    #     r[ωxmap[ie]] = 0
    #     r[ωymap[ie]] = 0
    #     r[χxmap[ie]] = 0
    #     r[χxmap[ie]] = 0
    # end

    # # bot b.c.
    # for i in eachindex(e.bot)
    #     ie = e.bot[i]
    #     push!(A, (ωxmap[ie], ωxmap[ie], 1))
    #     push!(A, (ωymap[ie], χymap[ie], 1))
    #     r[ωxmap[ie]] = 0
    #     r[ωymap[ie]] = 0
    # end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # top: dirichlet 
    A, r = add_dirichlet(A, r, ωxmap[bdy.sfc_nodes], 0)
    A, r = add_dirichlet(A, r, ωymap[bdy.sfc_nodes], 0)
    A, r = add_dirichlet(A, r, χxmap[bdy.sfc_nodes], 0)
    A, r = add_dirichlet(A, r, χymap[bdy.sfc_nodes], 0)

    # # bottom: dirichlet
    # A, r = add_dirichlet(A, r, ωxmap[bdy.bot_nodes], 0) 
    # A, r = add_dirichlet(A, r, ωymap[bdy.bot_nodes], χymap[bdy.bot_nodes], 0) # need to apply this on ωy since χy is full

    # special dirichlet conditions at z = -H:
    #              ∂x(χy) - ∂y(χx) = 0, 
    # -ε²*(∂x(ωx) - ∂y(ωy)) - β*χx = 0.
    A[ωxmap[bdy.bot_nodes], :] .= 0
    r[ωxmap[bdy.bot_nodes]] .= 0
    A[ωymap[bdy.bot_nodes], :] .= 0
    r[ωymap[bdy.bot_nodes]] .= 0
    w_quad, ξ_quad = quad_weights_points(2*g.order-1, 2)
    for k in axes(bdy.bot_tris, 1)
        # get tet associated with this bdy tri 

        # transform triangle to standard tri in x-y plane

        # compute ∫ φᵢ*∂x(φⱼ) dx dy,  ∫ φᵢ*∂y(φⱼ) dx dy, and ∫ φᵢ*φⱼ dx dy
        # for i's on the triangle and all j's in the tetrahedra

        # put the results (time 1, ε², or β as needed) as coefficients with the proper terms

        # get local indices of each point on edge `ie`:
    end
    println(@sprintf("%.1f s", time() - t₀))

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    println(@sprintf("%.1f s", time() - t₀))

    # reshape to get ω and χ
    ωx.values[:] = sol[ωxmap]
    ωy.values[:] = sol[ωymap]
    χx.values[:] = sol[χxmap]
    χy.values[:] = sol[χymap]
    return ωx, ωy, χx, χy
end

function pg_vort_res(; nref, order, showplots=false)
    # Ekman number
    ε² = 1
    println(@sprintf("q⁻¹ = %1.1e", sqrt(2*ε²)))

    # setup FE grids

    # gfile = "../meshes/bowl3D/mesh$nref.h5"
    # g  = FEGrid(gfile, order)
    # g1 = FEGrid(gfile, 1)

    gfile = "../meshes/mesh.h5"
    file = h5open(gfile, "r")
    p = read(file, "pts")
    t = read(file, "tets")
    tris_bot = read(file, "tris_bot")
    tris_sfc = read(file, "tris_sfc")
    bdy_bot = read(file, "bdy_bot")
    bdy_sfc = read(file, "bdy_sfc")
    e = unique!(vcat(bdy_bot, bdy_sfc))
    close(file)
    g = FEGrid(p, t, e, order)
    g1 = FEGrid(p, t, e, 1)
    println(@sprintf("h   = %1.1e", 1/cbrt(g.np)))

    # top and bottom surfaces
    ebot, etop = get_sides(g)
    bdy = (bot_nodes = ebot, sfc_nodes = etop, bot_tris = tris_bot, sfc_tris = tris_sfc) 

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    # forcing
    H(x, y) = sqrt(2 - x^2 - y^2) - 1
    x = g.p[:, 1] 
    y = g.p[:, 2] 
    z = g.p[:, 3] 
    δ = 0.1
    b = @. z + δ*exp(-(z + H(x, y))/δ)

    # initialize FE fields
    ωx = FEField(zeros(g.np), g, g1)
    ωy = FEField(zeros(g.np), g, g1)
    χx = FEField(zeros(g.np), g, g1)
    χy = FEField(zeros(g.np), g, g1)
    b  = FEField(b,           g, g1)

    # solve 
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, b, J, s, bdy, ε²)

    if showplots
        write_vtk(g, "../output/pg_vort", ["ωx"=>ωx, "ωy"=>ωy, "χx"=>χx, "χy"=>χy])
        println("../output/pg_vort.vtu")
    end

    return ωx, ωy, χx, χy
end

ωx, ωy, χx, χy = pg_vort_res(nref=0, order=2, showplots=true)

println("Done.")