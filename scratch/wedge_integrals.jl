using nuPGCM
using Printf

function test_quad()
    f(x, α, β, γ) = x[1]^α*x[2]^β*x[3]^γ

    w = Wedge(order=1)

    # p = w.p
    # truth=[1/2,1/4,1/6,1/8,1/10,1/6,1/12,1/18,1/24,1/12,1/24,1/36,1/20,1/40,1/30,1/6,1/12,1/18,1/24,1/24,1/48,1/72,1/60,1/120,1/120,1/12,1/24,1/36,1/60,1/120,1/180,1/20,1/40,1/120,1/30]
    p = [-1  1  0
          0  0  0
          1  1  0
         -1  1  1
          0  0  1 
          1  1  1]
    truth = [1,1/2,1/3,1/4,1/5,2/3,1/3,2/9,1/6,1/2,1/4,1/6,2/5,1/5,1/3,0,0,0,0,0,0,0,0,0,0,1/6,1/12,1/18,2/15,1/15,1/9,0,0,0,1/15]
    J = Jacobians(w, p, [1 2 3 4 5 6])
    wts = w.quad_wts*J.dets[1]
    x_pts = zeros(size(w.quad_pts))
    for i ∈ axes(x_pts, 1)
        x_pts[i, :] = x(w, w.quad_pts[i, :], p)
    end

    N = 4
    i = 1
    for α=0:N, β=0:N-α, γ=0:N-α-β
        ∫ = sum(wts[i]*f(x_pts[i, :], α, β, γ) for i ∈ axes(x_pts, 1))
        println(@sprintf("%d %d %d % 1.5f % 1.5f %1.16f", α, β, γ, ∫, truth[i], abs(∫ - truth[i])))
        i += 1
    end
end 

test_quad()