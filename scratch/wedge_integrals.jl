using nuPGCM
using Printf

function test_advection(h)
    el_H = Triangle(order=2)
    p_sfc = h*el_H.p_ref
    t_sfc = [1 2 3 4 5 6]
    e_sfc = []
    g_sfc = Grid(2, p_sfc, t_sfc, e_sfc)
    # H = FEField([0, h, h, h/2, h, h/2], g_sfc)
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

function test_quad()
    f(x, α, β, γ) = x[1]^α*x[2]^β*x[3]^γ

    w = Wedge(order=1)
    wts, pts = quad_weights_points(w)

    # p = w.p_ref
    # truth=[1/2,1/4,1/6,1/8,1/10,1/6,1/12,1/18,1/24,1/12,1/24,1/36,1/20,1/40,1/30,1/6,1/12,1/18,1/24,1/24,1/48,1/72,1/60,1/120,1/120,1/12,1/24,1/36,1/60,1/120,1/180,1/20,1/40,1/120,1/30]
    p = [-1  1  0
         0  0  0
         1  1  0
         -1  1  1
         0  0  1 
         1  1  1]
    truth = [1,1/2,1/3,1/4,1/5,2/3,1/3,2/9,1/6,1/2,1/4,1/6,2/5,1/5,1/3,0,0,0,0,0,0,0,0,0,0,1/6,1/12,1/18,2/15,1/15,1/9,0,0,0,1/15]
    js = [j(w, pts[i, :], p) for i ∈ axes(pts, 1)]
    wts = wts.*js
    x_pts = zeros(size(pts))
    for i ∈ axes(pts, 1)
        x_pts[i, :] = x(w, pts[i, :], p)
    end

    N = 4
    i = 1
    for α=0:N, β=0:N-α, γ=0:N-α-β
        ∫ = sum(wts[i]*f(x_pts[i, :], α, β, γ) for i ∈ axes(pts, 1))
        println(@sprintf("%d %d %d % 1.5f % 1.5f %1.16f", α, β, γ, ∫, truth[i], abs(∫ - truth[i])))
        i += 1
    end
end 

# test_quad()

# for h=[1, 0.1, 0.01]
#     A = test_advection(h)
#     println(maximum(abs.(A)))
# end