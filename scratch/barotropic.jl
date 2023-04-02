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

function solve_barotropic(g, r_sym, r_asym, β, τx, τy, ωx_bot, ωy_bot)
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
            φi = φ(g.sf, i, ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_sym([x, y], k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*∂x∂ξ
            # return -ε²*r_sym([x, y], k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*H(x, y)^2*∂x∂ξ - 
            #         ε²*r_sym([x, y], k)*(φi*∂xφ_j*Hx(x, y) + φi*∂yφ_j*Hy(x, y))*2*H(x, y)*∂x∂ξ 
        end
        K = [nuPGCM.ref_el_quad(ξ -> func_K(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # K′
        function func_K′(ξ, i, j)
            x, y = T(ξ)
            φi = φ(g.sf, i, ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_asym([x, y], k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*∂x∂ξ
            # return -ε²*r_asym([x, y], k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*H(x, y)^2*∂x∂ξ -
            #         ε²*r_asym([x, y], k)*(φi*∂xφ_j*Hx(x, y) - φi*∂yφ_j*Hy(x, y))*2*H(x, y)*∂x∂ξ 
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
            # return ((H(x, y)*β - f*Hy(x, y))*∂xφ_j + f*Hx(x, y)*∂yφ_j)*φi*∂x∂ξ
        end
        C = [nuPGCM.ref_el_quad(ξ -> func_C(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # rhs
        function func_r(ξ, i)
            x, y = T(ξ)
            τ_curl = (∂x(τy, [x, y], k) - ∂y(τx, [x, y], k))/H(x, y) - (τy([x, y], k)*Hx(x, y) - τx([x, y], k)*Hy(x, y))/H(x, y)^2
            ω_bot_div = ∂x(ωx_bot, [x, y], k) + ∂y(ωy_bot, [x, y], k)
            # τ_curl = (∂x(τy, [x, y], k) - ∂y(τx, [x, y], k))*H(x, y) - (τy([x, y], k)*Hx(x, y) - τx([x, y], k)*Hy(x, y))
            # ω_bot_div = (∂x(ωx_bot, [x, y], k) + ∂y(ωy_bot, [x, y], k))*H(x, y)^2
            φi = φ(g.sf, i, ξ)
            return ε²*(τ_curl + ω_bot_div)*φi*∂x∂ξ
        end
        r = [nuPGCM.ref_el_quad(ξ -> func_r(ξ, i), quad_wts, quad_pts) for i=1:g.nn]

        # interior terms
        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ∉ bdy 
                push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], K′[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], C[i, j]))
            end
        end
        rhs[g.t[k, :]] += r
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
        rhs[i] = 0
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # solve
    println("Solving...")
    return FEField(A\rhs, g)
end

function main(; order)
    # surface mesh
    g_sfc = FEGrid("meshes/circle/mesh2.h5", order)
    x = g_sfc.p[:, 1]
    y = g_sfc.p[:, 2]

    quick_plot(FEField(H.(x, y), g_sfc), L"H", "scratch/images/H.png")
    quick_plot(FEField(Hx.(x, y), g_sfc), L"H_x", "scratch/images/Hx.png")
    quick_plot(FEField(Hy.(x, y), g_sfc), L"H_y", "scratch/images/Hy.png")

    β = 1
    f(y) = β*y
    f_over_H = @. f(y)/(H(x, y) + 1e-5)
    quick_plot(FEField(f_over_H, g_sfc), L"f/H", "scratch/images/f_over_H.png", vmax=6)

    # wind stress
    τx = FEField(-cos.(π*y), g_sfc)
    τy = FEField(zeros(g_sfc.np), g_sfc)
    f_curl(x, y) = (∂x(τy, [x, y]) - ∂y(τx, [x, y]))*H(x, y) - (τy([x, y])*Hx(x, y) - τx([x, y])*Hy(x, y))
    # f_curl(x, y) = -π*sin(π*y)*H(x, y) - cos(π*y)*Hy(x, y)
    curl = f_curl.(x, y) 
    quick_plot(FEField(curl, g_sfc), L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", "scratch/images/curl.png")

    # 3D mesh
    g, el_cols, node_cols, p_to_tri = gen_3D_valign_mesh(g_sfc, H, order=1)

    # get ω's
    ωx_Ux_bot, ωy_Ux_bot, ωx_Uy_bot, ωy_Uy_bot = get_ω_U(g_sfc, g, node_cols, H, ε², f)
    quick_plot(FEField(ωy_Ux_bot.values./H.(x, y).^2, g_sfc), L"\omega^y_{U^x}(-H)/H^2", "scratch/images/omegax_Ux_H2.png")
    r_sym = ωy_Ux_bot/FEField(H.(x, y), g_sfc)^3
    r_asym = ωx_Ux_bot/FEField(H.(x, y), g_sfc)^3
    # r_sym = FEField(1e-1./H.(x, y), g_sfc)
    # r_asym = FEField(0, g_sfc)
    quick_plot(r_sym, L"r_\mathrm{sym}", "scratch/images/r_sym.png")
    quick_plot(r_asym, L"r_\mathrm{asym}", "scratch/images/r_asym.png")

    ωx_τx_bot, ωy_τx_bot, ωx_τy_bot, ωy_τy_bot = get_ω_τ(g_sfc, g, node_cols, ε², f)
    ωx_bot = (τx*ωx_τx_bot + τy*ωx_τy_bot)/FEField(H.(x, y), g_sfc)
    ωy_bot = (τx*ωy_τx_bot + τy*ωy_τy_bot)/FEField(H.(x, y), g_sfc)
    quick_plot(ωx_bot, L"\tau^j \omega^x_{\tau^j} / H", "scratch/images/omegax_bot.png")
    quick_plot(ωy_bot, L"\tau^j \omega^y_{\tau^j} / H", "scratch/images/omegay_bot.png")
    # ωx_bot = FEField(0, g_sfc)
    # ωy_bot = FEField(0, g_sfc)

    Ψ = solve_barotropic(g_sfc, r_sym, r_asym, β, τx, τy, ωx_bot, ωy_bot)
    quick_plot(Ψ, L"\Psi", "scratch/images/psi.png")
end

ε² = 1e-2
H(x, y) = 1 - x^2 - y^2
Hx(x, y) = -2x
Hy(x, y) = -2y

main(order=1)