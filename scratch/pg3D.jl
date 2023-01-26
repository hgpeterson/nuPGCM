using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

Line2D = pyimport("matplotlib.lines").Line2D
plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    ux, uy, uz, p = solve_pg3D(ux, uy, uz, p, J, s, bdy, f, ε²)

PG:
     -ε²∂zz(uˣ) - uʸ + ∂x(p) = fˣ  on Ω,
     -ε²∂zz(uʸ) + uˣ + ∂y(p) = fʸ  on Ω,
                       ∂z(p) = fᶻ  on Ω,
    ∂x(uˣ) + ∂y(uʸ) + ∂z(uᶻ) = 0   on Ω,
"""
function solve_pg3D(ux, uy, uz, p, J, s, bdy, f, ε²)
    # indices
    uxmap = 1:ux.g.np
    uymap = uxmap[end] .+ (1:uy.g.np)
    uzmap = uymap[end] .+ (1:uz.g.np)
    pmap  = uzmap[end] .+ (1:p.g.np)
    N = pmap[end]
    println("N = $N")

    # tag whether node of tet is on boundary or not
    on_ubot = [ux.g.t[k, i] ∈ bdy.ubot for k ∈ axes(ux.g.t, 1), i ∈ axes(ux.g.t, 2)]
    on_wbot = [uz.g.t[k, i] ∈ bdy.wbot for k ∈ axes(uz.g.t, 1), i ∈ axes(uz.g.t, 2)]
    on_wsfc = [uz.g.t[k, i] ∈ bdy.wsfc for k ∈ axes(uz.g.t, 1), i ∈ axes(uz.g.t, 2)]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)
    print("Building... ")
    t₀ = time()
    for k=1:ux.g.nt
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        Kᵏ = J.dets[k]*sum(s.uu.K.*JJ, dims=(1, 2))[1, 1, :, :]
        Mᵏ = J.dets[k]*s.uu.M
        Cxᵏ = J.dets[k]*sum(s.pu.CT.*J.Js[k, :, 1], dims=1)[1, :, :]
        Cyᵏ = J.dets[k]*sum(s.pu.CT.*J.Js[k, :, 2], dims=1)[1, :, :]
        Czᵏ = J.dets[k]*sum(s.pw.CT.*J.Js[k, :, 3], dims=1)[1, :, :]

        # RHSs
        b[uxmap[ux.g.t[k, :]]] .+= Mᵏ*f.fx.values[f.fx.g.t[k, :]]
        b[uymap[uy.g.t[k, :]]] .+= Mᵏ*f.fy.values[f.fy.g.t[k, :]]
        b[uzmap[uz.g.t[k, :]]] .+= J.dets[k]*s.ww.M*f.fz.values[f.fz.g.t[k, :]]

        # ∂z(u)*∂z(v)
        for i=1:ux.g.nn, j=1:ux.g.nn
            if on_ubot[k, i] continue end
            push!(A, (uxmap[ux.g.t[k, i]], uxmap[ux.g.t[k, j]], ε²*Kᵏ[i, j]))
        end
        for i=1:uy.g.nn, j=1:uy.g.nn
            if on_ubot[k, i] continue end
            push!(A, (uymap[uy.g.t[k, i]], uymap[uy.g.t[k, j]], ε²*Kᵏ[i, j]))
        end
        # u*v
        for i=1:ux.g.nn, j=1:uy.g.nn
            if on_ubot[k, i] continue end
            push!(A, (uxmap[ux.g.t[k, i]], uymap[uy.g.t[k, j]], -Mᵏ[i, j]))
        end
        for i=1:uy.g.nn, j=1:ux.g.nn
            if on_ubot[k, i] continue end
            push!(A, (uymap[uy.g.t[k, i]], uxmap[ux.g.t[k, j]], Mᵏ[i, j]))
        end
        # -p*(∇⋅v) term
        for i=1:ux.g.nn, j=1:p.g.nn
            if on_ubot[k, i] continue end
            push!(A, (uxmap[ux.g.t[k, i]], pmap[p.g.t[k, j]], -Cxᵏ[i, j]))
        end
        for i=1:uy.g.nn, j=1:p.g.nn
            if on_ubot[k, i] continue end
            push!(A, (uymap[uy.g.t[k, i]], pmap[p.g.t[k, j]], -Cyᵏ[i, j]))
        end
        for i=1:uz.g.nn, j=1:p.g.nn
            if on_wbot[k, i] || on_wsfc[k, i] continue end
            push!(A, (uzmap[uz.g.t[k, i]], pmap[p.g.t[k, j]], -Czᵏ[i, j]))
        end
        # q*(∇⋅u) term 
        for i=1:p.g.nn, j=1:ux.g.nn
            push!(A, (pmap[p.g.t[k, i]], uxmap[ux.g.t[k, j]], Cxᵏ[j, i]))
        end
        for i=1:p.g.nn, j=1:uy.g.nn
            push!(A, (pmap[p.g.t[k, i]], uymap[uy.g.t[k, j]], Cyᵏ[j, i]))
        end
        for i=1:p.g.nn, j=1:uz.g.nn
            push!(A, (pmap[p.g.t[k, i]], uzmap[uz.g.t[k, j]], Czᵏ[j, i]))
        end
    end

    # dirichlet b.c.
    for i ∈ bdy.ubot
        push!(A, (uxmap[i], uxmap[i], 1))
        push!(A, (uymap[i], uymap[i], 1))
        b[uxmap[i]] = 0
        b[uymap[i]] = 0
    end
    for i ∈ bdy.wbot
        push!(A, (uzmap[i], uzmap[i], 1))
        b[uzmap[i]] = 0
    end
    for i ∈ bdy.wsfc
        push!(A, (uzmap[i], uzmap[i], 1))
        b[uzmap[i]] = 0
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # set p to zero somewhere
    A, b = add_dirichlet(A, b, pmap[1], 0)
    println(time() - t₀, " s")

    # println("rank(A) = ", rank(A))

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\b
    println(time() - t₀, " s")

    # reshape to get u and p
    ux.values[:] = sol[uxmap]
    uy.values[:] = sol[uymap]
    uz.values[:] = sol[uzmap]
    p.values[:] = sol[pmap]
    return ux, uy, uz, p
end

function pg3D_res(; nref, plot=false)
    # Ekman number
    ε² = 1

    # get grids
    gfile = "../meshes/bowl3D/mesh$nref.h5"
    # gfile = "../meshes/valign3D/mesh$nref.h5"
    gu = FEGrid(gfile, 2)
    gw = FEGrid(gfile, 1)
    gp = FEGrid(gfile, 0)
    g1 = FEGrid(gfile, 1)

    # boundary struct
    ubot, usfc = get_sides(gu)
    wbot, wsfc = get_sides(gw)
    bdy = (ubot=ubot, usfc=usfc, wbot=wbot, wsfc=wsfc) 

    # cells = [MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, gu.t[i, :]) for i ∈ axes(gu.t, 1)]
    # vtk_grid("../output/bdy.vtu", gu.p', cells) do vtk
    #     bot = zeros(gu.np)
    #     bot[bot_nodes] .= 1
    #     sfc = zeros(gu.np)
    #     sfc[sfc_nodes] .= 1
    #     vtk["bot"] = bot
    #     vtk["sfc"] = sfc
    # end

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    ww = ShapeFunctionIntegrals(gw.s, gw.s)
    pu = ShapeFunctionIntegrals(gp.s, gu.s)
    pw = ShapeFunctionIntegrals(gp.s, gw.s)
    pp = ShapeFunctionIntegrals(gp.s, gp.s)
    s = (uu=uu,
         ww=ww,
         pu=pu,
         pw=pw,
         pp=pp)
         
    # forcing
    fx = FEField(zeros(gu.np), gu, g1)
    fy = FEField(zeros(gu.np), gu, g1)
    # fz = FEField(gw.p[:, 1], gw, g1)
    x = gw.p[:, 1] 
    y = gw.p[:, 2] 
    z = gw.p[:, 3] 
    H = @. sqrt(2 - x^2 - y^2) - 1
    δ = 0.1
    b = @. z + δ*exp(-(z + H)/δ)
    fz = FEField(b, gw, g1)
    f = (fx=fx, fy=fy, fz=fz)

    # get Jacobians
    J = Jacobians(g1)

    # initialize FE fields
    ux  = FEField(zeros(gu.np), gu, g1)
    uy  = FEField(zeros(gu.np), gu, g1)
    uz  = FEField(zeros(gw.np), gw, g1)
    p   = FEField(zeros(gp.np), gp, g1)

    # solve 
    ux, uy, uz, p = solve_pg3D(ux, uy, uz, p, J, s, bdy, f, ε²)

    if plot
        write_vtk(gu, "../output/pg3D_u", ["ux"=>ux, "uy"=>uy])
        write_vtk(gw, "../output/pg3D_w", ["uz"=>uz])
        write_vtk(gp, "../output/pg3D_p", ["p"=>p])
        println("../output/pg3D_*.vtu")
    end
end

pg3D_res(nref=2, plot=true)

println("Done.")