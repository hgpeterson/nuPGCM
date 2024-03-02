using nuPGCM
using PyPlot
using LinearAlgebra
using Random
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

Random.seed!(42)

"""
q = ∇²ψ + βy - ψ/λ²
"""
function build_inversion_LHS(g, K, M, λ)
    A = -K - M/λ^2
    for i ∈ g.e["bdy"]
        A[i, :] .= 0
        A[i, i] = 1.
    end
    return lu(A)
end

function invert!(ψ, LHS, M, q)
    RHS = M*q.values
    RHS[ψ.g.e["bdy"]] .= 0
    ψ.values[:] = LHS\RHS
    return ψ
end

"""
u = -∂y(ψ)
v =  ∂x(ψ)
"""
function compute_velocities(ψ)
    u = -∂(ψ, 2)
    v = ∂(ψ, 1)
    return FVField(u), FVField(v)
end

"""
Aᵢⱼₖ = ∫ [∂x(ψⱼ)∂y(qₖ) - ∂y(ψⱼ)∂x(qₖ)] φᵢ. 
"""
function build_advection_array(g)
    # unpack
    J = g.J
    el = g.el

    # compute general integrals
    f(ξ, i, j, k, d1, d2) = ∂φ(el, ξ, k, d1)*∂φ(el, ξ, j, d2)*φ(el, ξ, i)
    A_el = [nuPGCM.ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), el) for i=1:el.n, j=1:el.n, k=1:el.n, d1=1:2, d2=1:2]

    # allocate
    A = zeros(g.nt, el.n, el.n, el.n)

    for k=1:g.nt
        # unpack
        jac = J.Js[k, :, :]
        Δ = J.dets[k]

        for d1=1:2, d2=1:2
            A[k, :, :, :] += A_el[:, :, :, d1, d2]*(jac[d1, 2]*jac[d2, 1] - jac[d1, 1]*jac[d2, 2])*Δ
        end
    end

    return A
end

function build_SD_matrices(g, δ, u, v)
    J = g.J
    el = g.el
    M_SD = zeros(g.np, g.np)
    A_SD = zeros(g.np, g.np)
    for k ∈ 1:g.nt
        jac = J.Js[k, :, :]
        Δ = J.dets[k]

        f_M(ξ, i, j) = (u(ξ, k)*(∂φ(el, ξ, i, 1)*jac[1, 1] + ∂φ(el, ξ, i, 2)*jac[2, 1]) + 
                        v(ξ, k)*(∂φ(el, ξ, i, 1)*jac[1, 2] + ∂φ(el, ξ, i, 2)*jac[2, 2]))*φ(el, ξ, j)*Δ
        M_SDᵏ = [nuPGCM.ref_el_quad(ξ -> f_M(ξ, i, j), el) for i ∈ 1:g.nn, j ∈ 1:g.nn]

        f_A(ξ, i, j) = (u(ξ, k)*(∂φ(el, ξ, i, 1)*jac[1, 1] + ∂φ(el, ξ, i, 2)*jac[2, 1]) + 
                        v(ξ, k)*(∂φ(el, ξ, i, 1)*jac[1, 2] + ∂φ(el, ξ, i, 2)*jac[2, 2]))*
                       (u(ξ, k)*(∂φ(el, ξ, j, 1)*jac[1, 1] + ∂φ(el, ξ, j, 2)*jac[2, 1]) + 
                        v(ξ, k)*(∂φ(el, ξ, j, 1)*jac[1, 2] + ∂φ(el, ξ, j, 2)*jac[2, 2]))*Δ
        A_SDᵏ = [nuPGCM.ref_el_quad(ξ -> f_A(ξ, i, j), el) for i ∈ 1:g.nn, j ∈ 1:g.nn]

        for i ∈ 1:g.nn, j ∈ 1:g.nn
            M_SD[g.t[k, i], g.t[k, j]] += δ[k]*M_SDᵏ[i, j]
            A_SD[g.t[k, i], g.t[k, j]] += M_SDᵏ[j, i] + δ[k]*A_SDᵏ[i, j]
        end
    end
    return M_SD, A_SD
end

function advection(A, q, ψ)
    g = ψ.g
    adv = zeros(g.np)
    for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ ∈ 1:g.nn, iq ∈ 1:g.nn
        adv[g.t[k, i]] += A[k, i, iψ, iq]*ψ[g.t[k, iψ]]*q[g.t[k, iq]]
    end
    return adv
end

function evolve()
    # params
    Δt = 1e-2
    λ = 0.5

    # grid
    g = Grid(Triangle(order=1), "../meshes/circle/mesh3.h5")

    # δ = 0.1*(local mesh width)
    δ = 0.1*sqrt.(g.J.dets)*sqrt(3)/3

    # matrices
    K = nuPGCM.stiffness_matrix(g)
    M = nuPGCM.mass_matrix(g)
    adv_LHS = lu(M)
    inv_LHS = build_inversion_LHS(g, K, M, λ)
    A = build_advection_array(g)

    # initial condition
    # q = 0.5*randn(g.np)
    x = g.p[:, 1]
    y = g.p[:, 2]
    # Δ = 0.1
    # q = @. 0.9*(exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) - exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)))
    q = @. (1 - y^2)*(1 - x^2 - y^2)
    q = FEField(q, g)
    ψ = FEField(0, g)
    invert!(ψ, inv_LHS, M, q)
    u, v = compute_velocities(ψ)
    nuPGCM.quick_plot(ψ, cb_label=L"Streamfunction $\psi$", filename="$out_folder/psi.png")
    nuPGCM.quick_plot(q, cb_label=L"PV $q$", filename="$out_folder/q.png")
    nuPGCM.quick_plot(u, cb_label=L"u", filename="$out_folder/u.png")
    nuPGCM.quick_plot(v, cb_label=L"v", filename="$out_folder/v.png")

    # step forward
    t1 = time()
    N = 1700
    for i ∈ 1:N
        # invert
        invert!(ψ, inv_LHS, M, q)

        # advect (RK2)
        dq = adv_LHS\advection(A, q.values, ψ)
        dq = adv_LHS\advection(A, q.values + Δt/2*dq, ψ)
        q.values[:] = q.values - Δt*dq

        # # advect (RK2)
        # u, v = compute_velocities(ψ)
        # M_SD, A_SD = build_SD_matrices(g, δ, u, v)
        # dq = (M + M_SD)\(A_SD*q.values)
        # dq = (M + M_SD)\(A_SD*(q.values + Δt/2*dq))
        # q.values[:] = q.values - Δt*dq

        if mod(i, 100) == 0
            # CFL
            u, v = compute_velocities(ψ)
            println("\ni = $i/$N")
            println("CFL Δt: ", min(minimum(abs.(δ./u.values)), minimum(abs.(δ./v.values))))

            # plots
            nuPGCM.quick_plot(ψ, cb_label=L"Streamfunction $\psi$", filename="$out_folder/psi.png")
            nuPGCM.quick_plot(q, cb_label=L"PV $q$", filename="$out_folder/q.png")
            nuPGCM.quick_plot(u, cb_label=L"u", filename="$out_folder/u.png")
            nuPGCM.quick_plot(v, cb_label=L"v", filename="$out_folder/v.png")
        end
    end
    t2 = time()
    println((t2 - t1)/N)

    return q
end

q = evolve()

println("Done.")