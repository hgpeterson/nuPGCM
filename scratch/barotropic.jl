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
using Printf

include("utils.jl")
include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function solve_barotropic(g, r_sym, r_asym, ωx_bot, ωy_bot)
    # indices
    N = g.np

    # unpack
    bdy = g.e["bdy"]
    J = g.J
    # s = g.sfi

    # integration
    quad_wts, quad_pts = quad_weights_points(deg=7, dim=2)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    rhs = zeros(N)
    println("Building matrices...")
    for k=1:g.nt
        # Jacobian terms
        ξx = J.Js[k, 1, 1]
        ξy = J.Js[k, 1, 2]
        ηx = J.Js[k, 2, 1]
        ηy = J.Js[k, 2, 2]
        ∂x∂ξ = J.dets[k]

        # transformation from reference triangle
        T(ξ) = transform_from_ref_el(ξ, g.p[g.t[k, 1:3], :])

        # K
        function func_K(ξ, i, j)
            x = T(ξ)
            φi = φ(g.sf, i, ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_sym(x, k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*∂x∂ξ
            # return -ε²*r_sym([x, y], k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*H(x, y)^2*∂x∂ξ - 
            #         ε²*r_sym([x, y], k)*(φi*∂xφ_j*Hx(x, y) + φi*∂yφ_j*Hy(x, y))*2*H(x, y)*∂x∂ξ 
        end
        K = [nuPGCM.ref_el_quad(ξ -> func_K(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # K′
        function func_K′(ξ, i, j)
            x = T(ξ)
            φi = φ(g.sf, i, ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_asym(x, k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*∂x∂ξ
            # return -ε²*r_asym([x, y], k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*H(x, y)^2*∂x∂ξ -
            #         ε²*r_asym([x, y], k)*(φi*∂xφ_j*Hx(x, y) - φi*∂yφ_j*Hy(x, y))*2*H(x, y)*∂x∂ξ 
        end
        K′ = [nuPGCM.ref_el_quad(ξ -> func_K′(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # J(f/H, Ψ) term
        function func_C(ξ, i, j)
            x = T(ξ)
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            φi = φ(g.sf, i, ξ)
            return ((H(x)*fy(x) - f(x)*Hy(x))*∂xφ_j + f(x)*Hx(x)*∂yφ_j)*φi/H(x)^2*∂x∂ξ
            # return ((H(x, y)*β - f*Hy(x, y))*∂xφ_j + f*Hx(x, y)*∂yφ_j)*φi*∂x∂ξ
        end
        C = [nuPGCM.ref_el_quad(ξ -> func_C(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # rhs
        function func_r(ξ, i)
            x = T(ξ)
            JEBAR = (-γy(x)*Hx(x) + γx(x)*Hy(x))/H(x)^2
            τ_curl = (∂τ∂x(x)[2] - ∂τ∂y(x)[1])/H(x) - (τ(x)[2]*Hx(x) - τ(x)[1]*Hy(x))/H(x)^2
            ω_bot_div = ∂x(ωx_bot, x, k) + ∂y(ωy_bot, x, k)
            # τ_curl = (∂x(τy, [x, y], k) - ∂y(τx, [x, y], k))*H(x, y) - (τy([x, y], k)*Hx(x, y) - τx([x, y], k)*Hy(x, y))
            # ω_bot_div = (∂x(ωx_bot, [x, y], k) + ∂y(ωy_bot, [x, y], k))*H(x, y)^2
            φi = φ(g.sf, i, ξ)
            return (-JEBAR + τ_curl + ε²*ω_bot_div)*φi*∂x∂ξ
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

        # JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        # K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        # M = J.dets[k]*s.M
        # for i=1:g.nn, j=1:g.nn
        #     if g.t[k, i] ∉ bdy
        #         push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
        #     end
        # end
        # rhs[g.t[k, :]] += M*ones(g.nn)
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

function invert(g_sfc; showplots=false, nonzero_b=true)
    if showplots
        quick_plot(H, g_sfc, L"H", "scratch/images/H.png")
        quick_plot(Hx, g_sfc, L"H_x", "scratch/images/Hx.png")
        quick_plot(Hy, g_sfc, L"H_y", "scratch/images/Hy.png")
        f_over_H(x) = f(x)/(H(x) + 1e-5)
        quick_plot(f_over_H, g_sfc, L"f/H", "scratch/images/f_over_H.png", vmax=6)
        curl(x) = (∂τ∂x(x)[2] - ∂τ∂y(x)[1])*H(x) - (τ(x)[2]*Hx(x) - τ(x)[1]*Hy(x))
        quick_plot(curl, g_sfc, L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", "scratch/images/curl.png")
        quick_plot(γ, g_sfc, L"\gamma", "scratch/images/gamma.png")
        JEBAR(x) = γy(x)*Hx(x) - γx(x)*Hy(x)
        quick_plot(JEBAR, g_sfc, L"-H^2 J(1/H, \gamma)", "scratch/images/JEBAR.png")
    end

    # meshes
    # g, el_cols, node_cols, p_to_tri = gen_3D_valign_mesh(g_sfc, H, order=1)

    # get ω_U's
    ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = get_ω_U(g_sfc, g, node_cols, H, ε², f, showplots=showplots)
    ωx_Ux_bot = FEField(ωx_Ux[g.e["bot"]], g_sfc)
    ωy_Ux_bot = FEField(ωy_Ux[g.e["bot"]], g_sfc)
    r_sym = ωy_Ux_bot/FEField(H, g_sfc)^3
    r_asym = ωx_Ux_bot/FEField(H, g_sfc)^3
    # r_sym = FEField(1e1./H.(x, y), g_sfc)
    # r_asym = FEField(0, g_sfc)

    # get ω_τ's
    ωx_τx, ωy_τx, χx_τx, χy_τx = get_ω_τ(g_sfc, g, node_cols, ε², f, showplots=showplots)
    ωx_τx_bot = FEField(ωx_τx[g.e["bot"]], g_sfc)
    ωy_τx_bot = FEField(ωy_τx[g.e["bot"]], g_sfc)
    ωx_τy_bot = -ωy_τx_bot
    ωy_τy_bot = ωx_τx_bot

    # get ω_b's
    if nonzero_b
        ωx_b, ωy_b, χx_b, χy_b = get_ω_b(g_sfc, g, el_cols, node_cols, p_to_tri, ε², f, H, b, showplots=showplots)
        ωx_b_bot = FEField(ωx_b[g.e["bot"]], g_sfc)
        ωy_b_bot = FEField(ωy_b[g.e["bot"]], g_sfc)
    else
        ωx_b_bot = FEField(0, g_sfc)
        ωy_b_bot = FEField(0, g_sfc)
    end

    # combine
    τx = FEField(x -> τ(x)[1], g_sfc)
    τy = FEField(x -> τ(x)[2], g_sfc)
    ωx_bot = (ωx_b_bot + τx*ωx_τx_bot + τy*ωx_τy_bot)/FEField(H, g_sfc)
    ωy_bot = (ωy_b_bot + τx*ωy_τx_bot + τy*ωy_τy_bot)/FEField(H, g_sfc)
    if showplots
        quick_plot(ωx_bot*FEField(H, g_sfc), L"\omega^x_b + \tau^j \omega^x_{\tau^j}", "scratch/images/omegax_bot.png")
        quick_plot(ωy_bot*FEField(H, g_sfc), L"\omega^y_b + \tau^j \omega^y_{\tau^j}", "scratch/images/omegay_bot.png")
    end

    # solve
    Ψ = solve_barotropic(g_sfc, r_sym, r_asym, ωx_bot, ωy_bot)
    if showplots
        quick_plot(Ψ, L"\Psi", "scratch/images/psi.png")
    end

    return Ψ
end

function convergence()
    nrefs = 4
    g_hr = FEGrid(1, "meshes/circle/mesh$nrefs.h5")
    Ψ_hr = invert(g_hr, nonzero_b=true, showplots=true)
    err = zeros(nrefs)
    n = 1000
    θs = 2π*rand(n)
    rs = 0.9*sqrt.(rand(n))
    samples = [rs.*cos.(θs) rs.*sin.(θs)]
    for i=0:nrefs-1
        println("Refinement $i")
        g = FEGrid(1, "meshes/circle/mesh$i.h5")
        Ψ = invert(g, nonzero_b=true)
        err_vals = [abs(Ψ(samples[i, :]) - Ψ_hr(samples[i, :])) for i=1:n]
        err[i+1] = maximum(err_vals)
        println(@sprintf("%1.1e", err[i+1]))
    end
    return err
end

ε² = 1e-4
β = 1
δ = 0.1
H(x) = 1 - x[1]^2 - x[2]^2
Hx(x) = -2x[1]
Hy(x) = -2x[2]
f(x) = β*x[2]
fy(x) = β
b(x) = x[3] + δ*exp(-(x[3] + H(x))/δ)
γ(x) = -H(x)^3/3 - δ^2*(δ - H(x) - δ*exp(-H(x)/δ))
γx(x) = -Hx(x)*H(x)^2 - δ^2*Hx(x)*(exp(-H(x)/δ) - 1)
γy(x) = -Hy(x)*H(x)^2 - δ^2*Hy(x)*(exp(-H(x)/δ) - 1)
τ(x) = (-cos(π*x[2]), 0)
∂τ∂x(x) = (0, 0)
∂τ∂y(x) = (π*sin(π*x[2]), 0)
# τ(x) = (0, 0)
# ∂τ∂x(x) = (0, 0)
# ∂τ∂y(x) = (0, 0)

# g_sfc = FEGrid(1, "meshes/circle/mesh4.h5")
# g, el_cols, node_cols, p_to_tri = gen_3D_valign_mesh(g_sfc, H, order=1)
Ψ = invert(g_sfc, showplots=true, nonzero_b=false)
# Ψ = invert(g_sfc, showplots=true)

# err = convergence()
# display(log2.(err[1:end-1]./err[2:end]))

# no b:
# nref L2 error
# 0    3.0e0
# 1    1.4e0
# 2    2.6e-1
# 3    2.7e-2
# -> O(h^3) convergence ??

# only b:
# nref L2
# 0    1.1e-03
# 1    6.9e-04
# 2    2.1e-04
# 3    3.7e-05
# -> O(h^2.5) convergence ??

# no b
# nref  max error of samples
# 0     6.0e00
# 1     2.5e00
# 2     4.3e-01
# 3     5.3e-02
# -> O(h^3.0)

# only b:
# nref  max error of samples
# 0     2.6e-03
# 1     1.1e-03
# 2     2.3e-04
# 3     4.4e-05
# -> O(h^2.4)

println("Done.")