## Solve
##     -∂x(r_sym ∂x(Ψ)) - ∂y(r_sym ∂y(Ψ)) - 
##         ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) +
##             ∂y(f/H)∂x(Ψ) - ∂x(f/H)∂y(Ψ) 
##     =
##     -J(1/H, γ) + ε² z⋅(∇×τ/H) - ε² ∇⋅(ν*(ωb + τʲω_τⱼ)/H)
## with Ψ = 0 on boundary.

using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra

include("utils.jl")
include("baroclinic3D.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function solve_barotropic(g, r_sym, r_asym, β)
    # indices
    N = g.np

    # unpack
    bdy = g.e["bdy"]
    J = g.J
    s = g.sfi

    # integration
    quad_wts, quad_pts = quad_weights_points(deg=7, dim=2)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    rhs = zeros(N)
    @showprogress "Building matrices..." for k=1:g.nt
        # Jacobian terms
        ξx = J.Js[k, 1, 1]; ξy = J.Js[k, 1, 2]; ηx = J.Js[k, 2, 1]; ηy = J.Js[k, 2, 2]
        ∂x∂ξ = J.dets[k]

        # matrices
        # JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        # K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        # K′ = J.dets[k]*((ηx*ξy - ηy*ξx)*s.K[2, 1, :, :] - (ηx*ξy - ηy*ξx)*s.K[1, 2, :, :])

        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(ξ, g.p[g.t[k, 1:3], :])

        # K
        function func_K(ξ, i, j)
            x, y = T(ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return r_sym([x, y], k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*∂x∂ξ
        end
        K = [nuPGCM.ref_el_quad(ξ -> func_K(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # K′
        function func_K′(ξ, i, j)
            x, y = T(ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return r_asym([x, y], k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*∂x∂ξ
        end
        K′ = [nuPGCM.ref_el_quad(ξ -> func_K′(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # J(f/H, Ψ) term
        function func_C(ξ, i, j)
            x, y = T(ξ)
            f = β*y
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            φi = φ(g.sf, i, ξ)
            return ((H(x, y)*β - f*Hy(x, y))*∂xφ_j + f*Hx(x, y)*∂yφ_j)*φi/H(x, y)^2*∂x∂ξ
        end
        C = [nuPGCM.ref_el_quad(ξ -> func_C(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # interior terms
        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ∉ bdy 
                push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], K′[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], C[i, j]))
            end
        end

        # rhs
        r = s.M*curl.(g.p[g.t[k, :], 1], g.p[g.t[k, :], 2])*∂x∂ξ
        for i=1:g.nn
            if g.t[k, i] ∉ bdy 
                rhs[g.t[k, i]] += r[i]
            end
        end
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
        rhs[i] = 0
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # solve
    return FEField(A\rhs, g)
end

function main(; order)
    g = FEGrid("meshes/circle/mesh2.h5", order)
    # g = FEGrid("meshes/square/mesh2.h5", order)
    x = g.p[:, 1]
    y = g.p[:, 2]

    quick_plot(FEField(H.(x, y), g), L"H", "scratch/images/H.png")
    quick_plot(FEField(Hx.(x, y), g), L"H_x", "scratch/images/Hx.png")
    quick_plot(FEField(Hy.(x, y), g), L"H_y", "scratch/images/Hy.png")
    quick_plot(FEField(curl.(x, y), g), L"\mathbf{z} \cdot \nabla \times (\tau / H)", "scratch/images/curl.png")

    # r_sym, r_asym = get_ω_U(H, ε², g)
    r_sym = FEField(-1e-2./H.(x, y).^3, g)
    r_asym =FEField( 1e-4./H.(x, y).^3, g)

    β = 1

    Ψ = solve_barotropic(g, r_sym, r_asym, β)
    quick_plot(Ψ, L"\Psi", "scratch/images/psi.png")
end

ε² = 1

# H(x, y) = 1
# Hx(x, y) = 0
# Hy(x, y) = 0

H(x, y) = 1 - x^2 - y^2
Hx(x, y) = -2*x
Hy(x, y) = -2*y

# Δ = 0.2
# G(r) = 1 - exp(-r^2/(2Δ^2))
# Gr(r) = r/Δ^2*exp(-r^2/(2Δ^2))
# # H(x, y) = G(x + 1)*G(1 - x)*G(y + 1)*G(1 - y) + 0.1
# # Hx(x, y) = (Gr(x + 1)*G(1 - x) - G(x + 1)*Gr(1 - x))*G(y + 1)*G(1 - y)
# # Hy(x, y) = (Gr(y + 1)*G(1 - y) - G(y + 1)*Gr(1 - y))*G(x + 1)*G(1 - x)
# H(x, y) = G(1 - x^2 - y^2) + 0.1
# Hx(x, y) = -2x*Gr(1 - x^2 - y^2)
# Hy(x, y) = -2y*Gr(1 - x^2 - y^2)

# τˣ = -cos(πy), τʸ = 0
δ = 1e-2
curl(x, y) = ε²*(π*sin(π*y)/(H(x, y) + δ) + cos(π*y)*Hy(x, y)/(H(x, y) + δ)^2)

main(order=1)