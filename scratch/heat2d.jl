using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function get_M(g::Grid)
    J = g.J
    s = g.sfi
    M = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        Mᵏ = J.dets[k]*s.M 
        for i=1:g.nn, j=1:g.nn
            push!(M, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), g.np, g.np)
end

function get_K(g::Grid)
    J = g.J
    s = g.sfi
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        # JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        Kᵏ = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        for i=1:g.nn, j=1:g.nn
            push!(K, (g.t[k, i], g.t[k, j], -Kᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np)
end

function get_RHS(g::Grid, f)
    quad_wts, quad_pts = quad_weights_points(deg=7, dim=2)
    J = g.J
    rhs = zeros(g.np)
    for k=1:g.nt 
        ∂x∂ξ = J.dets[k]
        T(ξ) = transform_from_ref_el(ξ, g.p[g.t[k, 1:3], :])
        function fφi(ξ, i)
            φi = φ(g.sf, i, ξ)
            return f(T(ξ))*φi*∂x∂ξ
        end
        r = [nuPGCM.ref_el_quad(ξ -> fφi(ξ, i), quad_wts, quad_pts) for i=1:g.nn]
        rhs[g.t[k, :]] += r
    end
    for i ∈ g.e["bdy"]
        rhs[i] = 0
    end
    return rhs
end

function evolve()
    g = Grid(1, "meshes/valign2D/mesh3.h5")
    H(x) = 1 - x^2
    Hxx(x) = -2
    # H = @. 1 - g.p[:, 1]^2

    # g = Grid(2, "meshes/gmsh/mesh4.h5")
    # H = @. sqrt(2 - g.p[:, 1]^2) - 1

    # g = Grid(1, "meshes/square/mesh5.h5")
    # g.p[:, 2] = (g.p[:, 2] .- 1)/2
    # g = Grid(1, g.p, g.t, g.e)
    # H = ones(g.np)

    ε² = 1e-2
    μ = 1
    ϱ = 1e-4
    α = ε²/(μ*ϱ)
    T = 1e-2/α
    n_steps = 20
    Δt = T/n_steps

    x = g.p[:, 1]
    z = g.p[:, 2]
    # b = @. H*z^2 + 2/3*z^3
    b = @. z*(z + H(x))

    M = get_M(g)
    K = get_K(g)
    # LHS = lu(μ*ϱ*M - ε²*Δt/2*K)
    LHS = M - α*Δt/2*K
    LHS[g.e["bdy"], :] .= 0
    for i ∈ g.e["bdy"]
        LHS[i, i] = 1
    end
    LHS = lu(LHS)
    RHS = M + α*Δt/2*K

    for i=1:n_steps
        # b = LHS\(RHS*b)
        r = RHS*b
        r[g.e["bdy"]] .= 0
        # r_f = get_RHS(g, x -> (f(x, (i-1)*Δt, H, Hxx, α) + f(x, i*Δt, H, Hxx, α))/2)
        r_f = get_RHS(g, x -> f(x, (i-1)*Δt, H, Hxx, α))
        # r_f = zeros(g.np)
        b = LHS\(r + r_f)
    end

    # ba = [b_a(g.p[i, 2], T, ε²/μ/ϱ, H[i]) for i=1:g.np]
    ba = [b_a(g.p[i, :], T, H) for i=1:g.np]
    err = FEField(abs.(b - ba), g)
    println(@sprintf("Max Error: %1.1e at i=%d", maximum(err), argmax(err)))
    println(@sprintf("L2 Error: %1.1e", L2norm(err)))

    fig, ax, im = tplot(g.p, g.t, b, contour=true, cb_label=L"b")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/b2D.png")
    println("scratch/images/b2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, [f(g.p[i, :], T, H, Hxx, α) for i=1:g.np], contour=true, cb_label=L"f")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/f2D.png")
    println("scratch/images/f2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, ba, contour=true, cb_label=L"b_a")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/ba2D.png")
    println("scratch/images/ba2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, abs.(ba - b), cb_label="Error")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/error2D.png")
    println("scratch/images/error2D.png")
    plt.close()

    return b, ba
end

f(x, t, H, Hxx, α) = -(x[2]^2 + 2*α + x[2]*H(x[1]) + α*x[2]*Hxx(x[1]))*exp(-t)
b_a(x, t, H) = x[2]*(x[2] + H(x[1]))*exp(-t)

# """
# Analytical solution to ∂t(b) = α ∂zz(b) with ∂z(b) = 0 at z = -H, 0
# (truncated to Nth term in Fourier series).
# """
# function b_a(z, t, α, H; N=50)
#     if H == 0
#         return 0
#     end
#     # A(n) = 2*H*(1 + (-1)^(n+1))/(n^2*π^2)
#     # return -H/2 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
#     # A(n) = 8*H^3*(-1 + (-1)^n)/(n^4*π^4)
#     # return H^3/6 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
#     B(n) = 4*H^2*(1 + (-1)^(n+1))/(n*π)^3
#     return sum(B(n)*sin(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
# end

b, ba = evolve()
println("Done.")