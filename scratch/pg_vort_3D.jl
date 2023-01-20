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
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, f, diri, J, s, bdy, ε²)

Equations:
    -ε²∂zz(ωx) - (1 + β*y) ωy = f₁, 
    -ε²∂zz(ωy) + (1 + β*y) ωx = f₂,
                 ∂zz(χx) + ωx = f₃,
                 ∂zz(χy) + ωy = f₄,
Boundary conditions:
    ωx, ωy, χx, χy dirichlet at z = 0,
    ∂z(χx) = ∂z(χy) = 0  at  z = -H,
    χx, χy dirichlet at z = -H.
"""
function solve_pg_vort(ωx, ωy, χx, χy, f, diri, J, s, bdy, ε²)
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

        # RHSs
        r[ωxmap[g.t[k, :]]] += M*f.f1.values[g.t[k, :]]
        r[ωymap[g.t[k, :]]] += M*f.f2.values[g.t[k, :]]
        r[χxmap[g.t[k, :]]] += M*f.f3.values[g.t[k, :]]
        r[χymap[g.t[k, :]]] += M*f.f4.values[g.t[k, :]]

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

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # bottom: dirichlet
    A, r = add_dirichlet(A, r, ωxmap[bdy.bot_nodes], χxmap[bdy.bot_nodes], diri.χx_bot) 
    A, r = add_dirichlet(A, r, ωymap[bdy.bot_nodes], χymap[bdy.bot_nodes], diri.χy_bot) 
    # A, r = add_dirichlet(A, r, ωxmap[bdy.bot_nodes], diri.ωx_bot)
    # A, r = add_dirichlet(A, r, ωymap[bdy.bot_nodes], diri.ωy_bot)
    # A, r = add_dirichlet(A, r, χxmap[bdy.bot_nodes], diri.χx_bot)
    # A, r = add_dirichlet(A, r, χymap[bdy.bot_nodes], diri.χy_bot)

    # top: dirichlet 
    A, r = add_dirichlet(A, r, ωxmap[bdy.sfc_nodes], diri.ωx_sfc)
    A, r = add_dirichlet(A, r, ωymap[bdy.sfc_nodes], diri.ωy_sfc)
    A, r = add_dirichlet(A, r, χxmap[bdy.sfc_nodes], diri.χx_sfc)
    A, r = add_dirichlet(A, r, χymap[bdy.sfc_nodes], diri.χy_sfc)

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
    #         if all(j ∈ g.t[i, :] for j ∈ bdy.bot_tris[k_tri, :])
    #             k_tet = i
    #             break
    #         end
    #     end
    #     # println("Triangle $k_tri is in tetrahedron $k_tet.")

    #     # find which local indices of tetrahedron are on boundary
    #     il = findall(i -> g.t[k_tet, i] ∈ bdy.bot_tris[k_tri, :], 1:g.nn)

    #     # map bdy tri to ref tri in x-y plane
    #     x1 = g.p[bdy.bot_tris[k_tri, 1], :]
    #     x2 = g.p[bdy.bot_tris[k_tri, 2], :]
    #     x3 = g.p[bdy.bot_tris[k_tri, 3], :]
    #     area1 = norm(cross(x3-x1, x2-x1))

    #     # map ref tri in x-y plane to face of ref tet
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
    #     M = [sum(w_quad[k]*f_M(ξ_quad[k, :], i, j) for k ∈ eachindex(w_quad)) for i=il, j=1:g.nn]
    #     f_Cx(x, i, j) = φ(g.s, i, ξ(x))*(∂φ(g.s, j, 1, ξ(x))*ξx +  ∂φ(g.s, j, 2, ξ(x))*ηx)*area1*area2
    #     Cx = [sum(w_quad[k]*f_Cx(ξ_quad[k, :], i, j) for k ∈ eachindex(w_quad)) for i=il, j=1:g.nn]
    #     f_Cy(x, i, j) = φ(g.s, i, ξ(x))*(∂φ(g.s, j, 1, ξ(x))*ξy +  ∂φ(g.s, j, 2, ξ(x))*ηy)*area1*area2
    #     Cy = [sum(w_quad[k]*f_Cy(ξ_quad[k, :], i, j) for k ∈ eachindex(w_quad)) for i=il, j=1:g.nn]

    #     # χx = 0
    #     A[ωxmap[g.t[k_tet, il]], χxmap[g.t[k_tet, 1:g.nn]]] .+= M

    #     # χy = 0
    #     A[ωymap[g.t[k_tet, il]], χymap[g.t[k_tet, 1:g.nn]]] .+= M

    #     # # ∂x(χy) - ∂y(χx) = 0
    #     # A[ωymap[g.t[k_tet, il]], χymap[g.t[k_tet, 1:g.nn]]] .+= Cx
    #     # A[ωymap[g.t[k_tet, il]], χxmap[g.t[k_tet, 1:g.nn]]] .-= Cy

    #     # # -ε²*(∂x(ωx) + ∂y(ωy)) - β*χx = 0.
    #     # A[ωxmap[g.t[k_tet, il]], ωxmap[g.t[k_tet, 1:g.nn]]] .+= ε²*Cx
    #     # A[ωxmap[g.t[k_tet, il]], ωymap[g.t[k_tet, 1:g.nn]]] .+= ε²*Cy
    #     # A[ωxmap[g.t[k_tet, il]], χxmap[g.t[k_tet, 1:g.nn]]] .+= β*M

    #     # if you want something on the RHS other than 0
    #     x = g.p[g.t[k_tet, :], 1]
    #     y = g.p[g.t[k_tet, :], 2]
    #     z = g.p[g.t[k_tet, :], 3]
    #     r[ωxmap[g.t[k_tet, il]]] .+= M*(x.^2 .* exp.(y) .* z)
    #     r[ωymap[g.t[k_tet, il]]] .+= M*(x.^2 .* exp.(y) .* z)
    # end

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

    # setup FE grids
    gfile = "../meshes/valign3D/mesh$nref.h5"
    g  = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)
    println(@sprintf("h   = %1.1e", 1/cbrt(g.np)))

    # mesh resolution 
    h = 1/cbrt(g.np)

    # top and bottom nodes
    ebot, etop = get_sides(g)

    # surface triangles
    fmap, faces, bndix = all_faces(g.t)
    bdy_tris = unique(faces[bndix, :], dims=1)
    on_sfc = all(abs.(g.p[bdy_tris, 3]) .≤ 1e-4, dims=2)[:]
    sfc_tris = bdy_tris[on_sfc, :]
    bot_tris = bdy_tris[.!on_sfc, :]

    # boundary struct
    bdy = (bot_nodes=ebot, sfc_nodes=etop, bot_tris=bot_tris, sfc_tris=sfc_tris) 

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    # constructed solution
    x = g.p[:, 1] 
    y = g.p[:, 2] 
    z = g.p[:, 3] 
    H = @. 1 - x^2 - y^2
    ωx_a = @. x*exp(x*y*z)
    ωy_a = @. y*exp(x*y*z)
    χx_a = @. -(1 - H + exp(z)*(-1 + H + z))*cos(y)*sin(x)
    χy_a = @. -(1 - H + exp(z)*(-1 + H + z))*cos(x)*sin(y)
    diri = (ωx_bot=ωx_a[bdy.bot_nodes], ωx_sfc=ωx_a[bdy.sfc_nodes],
            ωy_bot=ωy_a[bdy.bot_nodes], ωy_sfc=ωy_a[bdy.sfc_nodes],
            χx_bot=χx_a[bdy.bot_nodes], χx_sfc=χx_a[bdy.sfc_nodes],
            χy_bot=χy_a[bdy.bot_nodes], χy_sfc=χy_a[bdy.sfc_nodes],
           )

    # forcing
    f1 = @. -y*exp(x*y*z)*(1 + ε²*x^3*y)
    f2 = @.  x*exp(x*y*z)*(1 - ε²*x*y^3)
    f3 = @. x*exp(x*y*z) + (-2*exp(z) - exp(z)*(-1 + H + z))*cos(y)*sin(x)
    f4 = @. y*exp(x*y*z) + (-2*exp(z) - exp(z)*(-1 + H + z))*cos(x)*sin(y)
    f1 = FEField(f1, g, g1)
    f2 = FEField(f2, g, g1)
    f3 = FEField(f3, g, g1)
    f4 = FEField(f4, g, g1)
    f = (f1=f1, f2=f2, f3=f3, f4=f4)

    # initialize FE fields
    ωx = FEField(zeros(g.np), g, g1)
    ωy = FEField(zeros(g.np), g, g1)
    χx = FEField(zeros(g.np), g, g1)
    χy = FEField(zeros(g.np), g, g1)

    # solve 
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, f, diri, J, s, bdy, ε²)

    if showplots
        write_vtk(g, "../output/pg_vort", ["ωx"=>ωx, "ωy"=>ωy, "χx"=>χx, "χy"=>χy])
        println("../output/pg_vort.vtu")

        ωx_a = FEField(ωx_a, g, g1)
        ωy_a = FEField(ωy_a, g, g1)
        χx_a = FEField(χx_a, g, g1)
        χy_a = FEField(χy_a, g, g1)
        # write_vtk(g, "../output/pg_vort_sol", ["ωx"=>ωx_a, "ωy"=>ωy_a, "χx"=>χx_a, "χy"=>χy_a])
        # println("../output/pg_vort_sol.vtu")
        # write_vtk(g, "../output/pg_vort_errs", ["ωx"=>abs(ωx - ωx_a), "ωy"=>abs(ωy - ωy_a), "χx"=>abs(χx - χx_a), "χy"=>abs(χy - χy_a)])
        # println("../output/pg_vort_errs.vtu")
    end

    err = L2norm(ωx - ωx_a, s, J) +
          L2norm(ωy - ωy_a, s, J) +
          L2norm(χx - χx_a, s, J) +
          L2norm(χy - χy_a, s, J)

    println(@sprintf("(h, err) = (%1.1e, %1.1e)", h, err))
    return h, err
end

h, err = pg_vort_res(nref=2, order=2, showplots=true)

# Errors: 

# setting χ through ω, neumann implied
# order = 1
# h       err
# 2.6e-1  2.2e-1
# 1.3e-1  1.1e-1
# 6.6e-2  6.0e-2
# 3.3e-2  3.4e-2
# ----------------> O(h)
# order = 2
# h       err
# 1.5e-1  1.8e-1
# 7.2e-2  5.0e-2
# 3.4e-2  1.3e-2
# ----------------> O(h^2)

# setting χ and ω normally, no neumann implied
# order = 1
# h       err
# 2.6e-1  2.8e-3
# 1.3e-1  1.4e-3
# 6.6e-2  7.6e-4
# ----------------> O(h)
# order = 2
# h       err
# 1.5e-1  1.1e-3
# 7.2e-2  2.0e-4
# 6.6e-2  4.1e-5
# ----------------> O(h^2)

println("Done.")