using nuPGCM
using Printf

function test_grid()
    el_H = Triangle(order=2)
    p_sfc = el_H.p_ref
    t_sfc = [1 2 3 4 5 6]
    e_sfc = Int64[]
    g_sfc = Grid(2, p_sfc, t_sfc, e_sfc)
    H = FEField([0, 0, 1, 1/2, 1/2, 0], g_sfc)
    return H
end

function test_advection(h)
    el_H = Triangle(order=2)
    p_sfc = h*el_H.p_ref
    t_sfc = [1 2 3 4 5 6]
    e_sfc = Int64[]
    g_sfc = Grid(2, p_sfc, t_sfc, e_sfc)
    H = FEField([0, 0, h, h/2, h/2, 0], g_sfc)

    el_χ = Wedge(order=1)
    el_b = Wedge(order=2)
    p = h*el_χ.p_ref
    wts, pts = quad_weights_points(el_χ)
    js = [j(el_χ, pts[i, :], p) for i ∈ axes(pts, 1)]
    wts .*= js

    f(ξ, i, j, k, d1, d2) = ∂φ(el_χ, ξ, k, d1)*∂φ(el_b, ξ, j, d2)*φ(el_b, ξ, i)/H(x(el_H, ξ[1:2], p_sfc), 1)
    return [nuPGCM.ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), wts, pts) for i=1:el_b.n, j=1:el_b.n, k=1:el_χ.n, d1=1:el_χ.dim, d2=1:el_χ.dim]
end

function test_barotropic(h)
    el_H = Triangle(order=2)
    p_sfc = h*el_H.p_ref
    t_sfc = [1 2 3 4 5 6]
    e_sfc = []
    g_sfc = Grid(2, p_sfc, t_sfc, e_sfc)
    H = FEField([0, 0, h, h/2, h/2, 0], g_sfc)

    el_Ψ = Triangle(order=1)
    p = h*el_Ψ.p_ref
    wts, pts = quad_weights_points(el_Ψ)
    js = [j(el_Ψ, pts[i, :], p) for i ∈ axes(pts, 1)]
    wts .*= js

    # f(ξ, i, j, d1, d2) = ∂φ(el_Ψ, ξ, i, d1)*∂φ(el_Ψ, ξ, j, d2)/H(x(el_H, ξ[1:2], p_sfc), 1)^3
    f(ξ, i, j, d1, d2) = φ(el_Ψ, ξ, i)*∂φ(el_Ψ, ξ, j, d2)/H(x(el_H, ξ[1:2], p_sfc), 1)
    return [nuPGCM.ref_el_quad(ξ -> f(ξ, i, j, d1, d2), wts, pts) for i=1:el_Ψ.n, j=1:el_Ψ.n, d1=1:el_Ψ.dim, d2=1:el_Ψ.dim]
end

function test()
    g = Grid(1, "meshes/circle/mesh0.h5")
    H = FEField(2, g)
    return H([0, 0])
end

println("Advection")
for h=[1, 0.1, 0.01]
    A = test_advection(h)
    println(maximum(abs.(A)))
end
println()

println("Barotropic")
for h=[1, 0.1, 0.01]
    A = test_barotropic(h)
    println(maximum(abs.(A)))
end
println()