using nuPGCM
using PyPlot
using LinearAlgebra

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

Random.seed!(42)

"""
∇²ψ - ψ/λ² = ζ
"""
function build_inversion_LHS(g, K, M, λ)
    A = -K - M/λ^2
    for i ∈ g.e["bdy"]
        A[i, :] .= 0
        A[i, i] = 1.
    end
    return lu(A)
end

function invert(g, LHS, M, ζ)
    RHS = M*ζ
    RHS[g.e["bdy"]] .= 0
    return LHS\RHS
end

"""
Aᵢⱼₖ = ∫ [∂x(ψⱼ)∂y(ζₖ) - ∂y(ψⱼ)∂x(ζₖ)] φᵢ. 
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

function advection(g, A, ζ, ψ)
    adv = zeros(g.np)
    for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ ∈ 1:g.nn, iζ ∈ 1:g.nn
        adv[g.t[k, i]] += A[k, i, iψ, iζ]*ψ[g.t[k, iψ]]*ζ[g.t[k, iζ]]
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
    t = 0
    # ζ = 0.5*randn(g.np)
    x = g.p[:, 1]
    y = g.p[:, 2]
    Δ = 0.1
    ζ = @. 0.9*(exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) - exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)))
    ψ = invert(g, inv_LHS, M, ζ)

    # step forward
    for i ∈ 1:10000
        # invert
        ψ = invert(g, inv_LHS, M, ζ)

        # advect (RK2)
        dζ = adv_LHS\advection(g, A, ζ, ψ)
        dζ = adv_LHS\advection(g, A, ζ + Δt/2*dζ, ψ)
        ζ -= Δt*dζ

        if mod(i, 100) == 0
            ψ = invert(g, inv_LHS, M, ζ)
            nuPGCM.quick_plot(FEField(ψ, g), cb_label=L"Streamfunction $\psi$", filename="$out_folder/psi.png")
            nuPGCM.quick_plot(FEField(ζ, g), cb_label=L"Vorticity $\zeta$", filename="$out_folder/zeta.png")
        end
    end
end

evolve()