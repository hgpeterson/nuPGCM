using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf
using HDF5

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, b, J, s, e, ε², β)

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
      -ε²*(∂x(τy) - ∂y(τx)) - ε²*(∂x(ωx) + ∂y(ωy)) - β*χx = 0  at  z = -H.
For now, we simplify the problem so that
    - f = 1,
    - τx = τy = 0, and
    - b.c.'s 4 and 5 are just χx = χy = 0 at z = -H.
"""
function solve_pg_vort(ωx, ωy, χx, χy, b, J, s, bdy, ε², β)
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
    # A, r = add_dirichlet(A, r, χxmap[bdy.sfc_nodes], g.p[bdy.sfc_nodes, 1])
    # A, r = add_dirichlet(A, r, χymap[bdy.sfc_nodes], g.p[bdy.sfc_nodes, 1])

    # # special dirichlet conditions at z = -H:
    # #              ∂x(χy) - ∂y(χx) = 0, 
    # # -ε²*(∂x(ωx) + ∂y(ωy)) - β*χx = 0.
    # A[ωxmap[bdy.bot_nodes], :] .= 0
    # r[ωxmap[bdy.bot_nodes]] .= 0
    # A[ωymap[bdy.bot_nodes], :] .= 0
    # r[ωymap[bdy.bot_nodes]] .= 0
    # w_quad, ξ_quad = quad_weights_points(2*g.order, 2)
    # ref = reference_element_nodes(1, 3)
    # for k_tri in axes(bdy.bot_tris, 1)
    #     # get tet associated with this bdy tri 
    #     k_tet = 0
    #     for i=1:g1.nt
    #         if sum(j ∈ g1.t[i, :] for j ∈ bdy.bot_tris[k_tri, :]) == 3
    #             k_tet = i
    #             break
    #         end
    #     end
    #     # println("Triangle $k_tri is in tetrahedron $k_tet.")

    #     # get indices of tetrahedron on boundary
    #     il = findall(i->g1.t[k_tet, i] in bdy.bot_tris[k_tri, :], 1:4)

    #     # bdy tri -> ref tri in x-y plane
    #     x1 = g.p[bdy.bot_tris[k_tri, 1], :]
    #     x2 = g.p[bdy.bot_tris[k_tri, 2], :]
    #     x3 = g.p[bdy.bot_tris[k_tri, 3], :]
    #     area1 = norm(cross(x3-x1, x2-x1))

    #     # ref tri in x-y plane to face of ref tet
    #     x1 = ref[il[1], :]
    #     x2 = ref[il[2], :]
    #     x3 = ref[il[3], :]
    #     ξ(x) = x1 + x[1]*(x2 - x1) + x[2]*(x3 - x1)
    #     area2 = norm(cross(x3-x1, x2-x1))

    #     # get ∂ξ/∂x, ∂ξ/∂y, ∂η/∂x, ∂η/∂y from J
    #     ξx = J.Js[k_tet, 1, 1]
    #     ξy = J.Js[k_tet, 1, 2]
    #     ηx = J.Js[k_tet, 2, 1]
    #     ηy = J.Js[k_tet, 2, 2]

    #     # compute ∫ φᵢ*∂x(φⱼ) dS,  ∫ φᵢ*∂y(φⱼ) dS, and ∫ φᵢ*φⱼ dS
    #     # for i's on the triangle and all j's in the tetrahedra
    #     f_M(x, i, j) = φ(g.s, i, ξ(x))*φ(g.s, j, ξ(x))*area1*area2
    #     M = [sum(w_quad[k]*f_M(ξ_quad[k, :], i, j) for k ∈ eachindex(w_quad)) for i=il, j=1:4]
    #     f_Cx(x, i, j) = φ(g.s, i, ξ(x))*(∂φ(g.s, j, 1, ξ(x))*ξx +  ∂φ(g.s, j, 2, ξ(x))*ηx)*area1*area2
    #     Cx = [sum(w_quad[k]*f_Cx(ξ_quad[k, :], i, j) for k ∈ eachindex(w_quad)) for i=il, j=1:4]
    #     f_Cy(x, i, j) = φ(g.s, i, ξ(x))*(∂φ(g.s, j, 1, ξ(x))*ξy +  ∂φ(g.s, j, 2, ξ(x))*ηy)*area1*area2
    #     Cy = [sum(w_quad[k]*f_Cy(ξ_quad[k, :], i, j) for k ∈ eachindex(w_quad)) for i=il, j=1:4]

    #     # χx = 0
    #     A[ωxmap[g.t[k_tet, il]], χxmap[g.t[k_tet, 1:4]]] .+= M

    #     # χy = 0
    #     A[ωymap[g.t[k_tet, il]], χymap[g.t[k_tet, 1:4]]] .+= M

    #     # # ∂x(χy) - ∂y(χx) = 0
    #     # A[ωymap[g.t[k_tet, il]], χymap[g.t[k_tet, 1:4]]] .+= Cx
    #     # A[ωymap[g.t[k_tet, il]], χxmap[g.t[k_tet, 1:4]]] .-= Cy

    #     # # -ε²*(∂x(ωx) + ∂y(ωy)) - β*χx = 0.
    #     # A[ωxmap[g.t[k_tet, il]], ωxmap[g.t[k_tet, 1:4]]] .+= ε²*Cx
    #     # A[ωxmap[g.t[k_tet, il]], ωymap[g.t[k_tet, 1:4]]] .+= ε²*Cy
    #     # A[ωxmap[g.t[k_tet, il]], χxmap[g.t[k_tet, 1:4]]] .+= β*M

    #     # if you want something on the RHS other than 0
    #     r[ωxmap[g.t[k_tet, il]]] .+= M*g.p[g.t[k_tet, 1:4], 3]
    #     r[ωymap[g.t[k_tet, il]]] .+= M*g.p[g.t[k_tet, 1:4], 3]
    # end

    # bottom: dirichlet
    # A, r = add_dirichlet(A, r, ωxmap[bdy.bot_nodes], 0) 
    A, r = add_dirichlet(A, r, ωxmap[bdy.bot_nodes], χxmap[bdy.bot_nodes], 0) 
    A, r = add_dirichlet(A, r, ωymap[bdy.bot_nodes], χymap[bdy.bot_nodes], 0) 

    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    if N < 10000
        R = rank(A)
        println("rank(A): ", R, " = N - ", N - R)
    end

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

    # beta-plane
    β = 1

    # setup FE grids
    gfile = "../meshes/valign3D/mesh$nref.h5"
    # gfile = "../meshes/bowl3D/mesh$nref.h5"
    g  = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)
    println(@sprintf("h   = %1.1e", 1/cbrt(g.np)))

    # top and bottom nodes
    ebot, etop = get_sides(g)

    # surface triangles
    fmap, faces, bndix = nuPGCM.all_faces(g1.t)
    bdy_tris = unique(faces[bndix, :], dims=1)
    on_sfc = (abs.(sum(g1.p[bdy_tris, 3], dims=2)) .≤ 1e-4)[:]
    sfc_tris = bdy_tris[on_sfc, :]
    bot_tris = bdy_tris[.!on_sfc, :]

    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, bot_tris[i, :]) for i in axes(bot_tris, 1)]
    vtk_grid("../output/bot.vtu", g1.p', cells) do vtk
    end
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, sfc_tris[i, :]) for i in axes(sfc_tris, 1)]
    vtk_grid("../output/sfc.vtu", g1.p', cells) do vtk
    end
    error()

    # boundary struct
    bdy = (bot_nodes=ebot, sfc_nodes=etop, bot_tris=bot_tris, sfc_tris=sfc_tris) 

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    # forcing
    x = g.p[:, 1] 
    y = g.p[:, 2] 
    z = g.p[:, 3] 
    # H(x, y) = sqrt(2 - x^2 - y^2) - 1
    # H(x, y) = 1 - x^2 - y^2
    # δ = 0.1
    # b = @. z + δ*exp(-(z + H(x, y))/δ)
    b = 2*x - 3*y

    # initialize FE fields
    ωx = FEField(zeros(g.np), g, g1)
    ωy = FEField(zeros(g.np), g, g1)
    χx = FEField(zeros(g.np), g, g1)
    χy = FEField(zeros(g.np), g, g1)
    b  = FEField(b,           g, g1)

    # solve 
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, b, J, s, bdy, ε², β)

    if showplots
        write_vtk(g, "../output/pg_vort", ["ωx"=>ωx, "ωy"=>ωy, "χx"=>χx, "χy"=>χy])
        println("../output/pg_vort.vtu")
    end

    return ωx, ωy, χx, χy
end

ωx, ωy, χx, χy = pg_vort_res(nref=2, order=2, showplots=true)

println("Done.")