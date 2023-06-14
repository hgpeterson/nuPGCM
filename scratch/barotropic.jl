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

"""
Solve
    -∂x(r_sym ∂x(Ψ)) - ∂y(r_sym ∂y(Ψ)) - 
        ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) +
            ∂y(f/H)∂x(Ψ) - ∂x(f/H)∂y(Ψ) 
    = -J(1/H, γ) + z⋅(∇×τ/H) - ε² ∇⋅(ν*ω_bot/H)
with Ψ = 0 on boundary.
"""
function solve_barotropic(g, r_sym, r_asym, ωx_τ_bot, ωy_τ_bot, ωx_b_bot, ωy_b_bot)
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
    print("Building matrices")
    t₀ = time()
    for k=1:g.nt
        if mod(k, Int64(round(0.25*g.nt))) == 0
            print(".")
        end
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
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_sym(x, k)*(∂xφ_i*∂xφ_j + ∂yφ_i*∂yφ_j)*∂x∂ξ
        end
        K = [nuPGCM.ref_el_quad(ξ -> func_K(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # K′
        function func_K′(ξ, i, j)
            x = T(ξ)
            ∂xφ_i = ∂φ(g.sf, i, 1, ξ)*ξx + ∂φ(g.sf, i, 2, ξ)*ηx
            ∂yφ_i = ∂φ(g.sf, i, 1, ξ)*ξy + ∂φ(g.sf, i, 2, ξ)*ηy
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            return -ε²*r_asym(x, k)*(∂xφ_i*∂yφ_j - ∂yφ_i*∂xφ_j)*∂x∂ξ
        end
        K′ = [nuPGCM.ref_el_quad(ξ -> func_K′(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # J(f/H, Ψ) term
        function func_C(ξ, i, j)
            x = T(ξ)
            ∂xφ_j = ∂φ(g.sf, j, 1, ξ)*ξx + ∂φ(g.sf, j, 2, ξ)*ηx
            ∂yφ_j = ∂φ(g.sf, j, 1, ξ)*ξy + ∂φ(g.sf, j, 2, ξ)*ηy
            φi = φ(g.sf, i, ξ)
            return ((H(x)*fy(x) - f(x)*Hy(x))*∂xφ_j + f(x)*Hx(x)*∂yφ_j)*φi/H(x)^2*∂x∂ξ
        end
        C = [nuPGCM.ref_el_quad(ξ -> func_C(ξ, i, j), quad_wts, quad_pts) for i=1:g.nn, j=1:g.nn]

        # rhs
        function func_r(ξ, i)
            x = T(ξ)
            JEBAR = (-γy(x)*Hx(x) + γx(x)*Hy(x))/H(x)^2
            τ_curl = (∂τ∂x(x)[2] - ∂τ∂y(x)[1])/H(x) - (τ(x)[2]*Hx(x) - τ(x)[1]*Hy(x))/H(x)^2
            ω_bot_div = ∂x(ωx_τ_bot, x, k) + ∂x(ωx_b_bot, x, k) + ∂y(ωy_τ_bot, x, k) + ∂y(ωy_b_bot, x, k)
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
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
        rhs[i] = 0
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    println(@sprintf(" (%.1f s)", time() - t₀))

    # solve
    return FEField(A\rhs, g)
end

function barotropic_inversion(g_sfc, g_cols, g, b_cols, z_cols, Dxs, Dys; showplots=false, nonzero_b=true)
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

    # get ω_U's
    ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = get_ω_U(g_sfc, g_cols, z_cols, H, ε², f, showplots=showplots)
    ωx_Ux_bot = zeros(g_sfc.np)
    ωy_Ux_bot = zeros(g_sfc.np)
    for i=1:g_sfc.np
        I = p_to_tri[i][1] # since ω_U's are actually continuous, just pick from one triangle
        k = I[1]
        j = I[2]
        ωx_Ux_bot[i] = ωx_Ux[k][g_cols[k].e["bot"][j]]
        ωy_Ux_bot[i] = ωy_Ux[k][g_cols[k].e["bot"][j]]
    end
    ωx_Ux_bot = FEField(ωx_Ux_bot, g_sfc)
    ωy_Ux_bot = FEField(ωy_Ux_bot, g_sfc)
    r_sym = ωy_Ux_bot/FEField(H, g_sfc)^3
    r_asym = ωx_Ux_bot/FEField(H, g_sfc)^3
    # r_sym = FEField(1e1./H.(x, y), g_sfc)
    # r_asym = FEField(0, g_sfc)

    # get ω_τ's
    ωx_τx, ωy_τx, χx_τx, χy_τx = get_ω_τ(g_sfc, g_cols, z_cols, H, ε², f, showplots=showplots)
    ωx_τx_bot = zeros(g_sfc.np)
    ωy_τx_bot = zeros(g_sfc.np)
    for i=1:g_sfc.np
        I = p_to_tri[i][1] # since ω_τ's are actually continuous, just pick from one triangle
        k = I[1]
        j = I[2]
        ωx_τx_bot[i] = ωx_τx[k][g_cols[k].e["bot"][j]]
        ωy_τx_bot[i] = ωy_τx[k][g_cols[k].e["bot"][j]]
    end
    ωx_τx_bot = FEField(ωx_τx_bot, g_sfc)/FEField(H, g_sfc)^2
    ωy_τx_bot = FEField(ωy_τx_bot, g_sfc)/FEField(H, g_sfc)^2
    ωx_τy_bot = -ωy_τx_bot
    ωy_τy_bot = ωx_τx_bot
    τx = FEField(x -> τ(x)[1], g_sfc)
    τy = FEField(x -> τ(x)[2], g_sfc)
    ωx_τ_bot = (τx*ωx_τx_bot + τy*ωx_τy_bot)/FEField(H, g_sfc)
    ωy_τ_bot = (τx*ωy_τx_bot + τy*ωy_τy_bot)/FEField(H, g_sfc)
    if showplots
        quick_plot(ωx_τ_bot*FEField(H, g_sfc), L"\omega^x_\tau(-H)", "scratch/images/omegax_tau_bot.png")
        quick_plot(ωy_τ_bot*FEField(H, g_sfc), L"\omega^y_\tau(-H)", "scratch/images/omegay_tau_bot.png")
    end

    # get ω_b's
    if nonzero_b
        ωx_b, ωy_b, χx_b, χy_b = get_ω_b(g_sfc, g_cols, b_cols, z_cols, Dxs, Dys, ε², f, b, showplots=showplots)
        ωx_b_bot = [ωx_b[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3]
        ωy_b_bot = [ωy_b[k][g_cols[k].e["bot"][i]] for k=1:g_sfc.nt, i=1:3]
        ωx_b_bot = DGField(ωx_b_bot, g_sfc)/DGField(H, g_sfc)
        ωy_b_bot = DGField(ωy_b_bot, g_sfc)/DGField(H, g_sfc)
    else
        ωx_b_bot = DGField(0, g_sfc)
        ωy_b_bot = DGField(0, g_sfc)
    end

    # solve
    Ψ = solve_barotropic(g_sfc, r_sym, r_asym, ωx_τ_bot, ωy_τ_bot, ωx_b_bot, ωy_b_bot)
    if showplots
        quick_plot(Ψ, L"\Psi", "scratch/images/psi.png")
    end

    return Ψ
end

function get_Ux_Uy(Ψ; showplots=false)
    g = Ψ.g
    Ux = FVField([-∂y(Ψ, [0, 0], k) for k=1:g.nt], g)
    Uy = FVField([+∂x(Ψ, [0, 0], k) for k=1:g.nt], g)
    if showplots
        quick_plot(Ux, L"U^x", "scratch/images/Ux.png")
        quick_plot(Uy, L"U^y", "scratch/images/Uy.png")
    end
    return Ux, Uy
end

ε² = 1e-2
δ = 0.1
H(x) = 1 - x[1]^2 - x[2]^2
Hx(x) = -2x[1]
Hy(x) = -2x[2]
# f(x) = 1 + x[2]
f(x) = 1
# fy(x) = 1
fy(x) = 0
b(x) = x[3] + δ*exp(-(x[3] + H(x))/δ)
γ(x) = -H(x)^3/3 - δ^2*(δ - H(x) - δ*exp(-H(x)/δ))
γx(x) = -Hx(x)*H(x)^2 - δ^2*Hx(x)*(exp(-H(x)/δ) - 1)
γy(x) = -Hy(x)*H(x)^2 - δ^2*Hy(x)*(exp(-H(x)/δ) - 1)
# τ(x) = (-cos(π*x[2]), 0)
# ∂τ∂x(x) = (0, 0)
# ∂τ∂y(x) = (π*sin(π*x[2]), 0)
τ(x) = (0, 0)
∂τ∂x(x) = (0, 0)
∂τ∂y(x) = (0, 0)

# # mesh
# geo = "circle"
# nref = 3
# g_sfc, g, g_cols, z_cols, p_to_tri = gen_3D_valign_mesh(geo, nref, H; chebyshev=true, tessellate=false)

# # second order b
# sf2 = ShapeFunctions(order=2, dim=3)
# sfi2 = ShapeFunctionIntegrals(sf2, sf2)
# b_cols = [Grid(2, col.p, col.t, col.e, sf2, sfi2) for col ∈ g_cols]

# # derivative matrices
# Dxs = Vector{Any}(undef, g_sfc.nt)
# Dys = Vector{Any}(undef, g_sfc.nt)
# @showprogress "Saving derivative matrices..." for k=1:g_sfc.nt
#     Dxs[k], Dys[k] = get_b_gradient_matrices(b_cols[k], g_cols[k], g_sfc, z_cols, k) 
# end

Ψ = barotropic_inversion(g_sfc, g_cols, g, b_cols, z_cols, Dxs, Dys, showplots=true, nonzero_b=true)
Ux, Uy = get_Ux_Uy(Ψ, showplots=true)

Hfield = FEField(H, g_sfc)
ωx_Ux, ωy_Ux, χx_Ux, χy_Ux = get_ω_U(g_sfc, g_cols, z_cols, H, ε², f, showplots=false)
ωx_b, ωy_b, χx_b, χy_b = get_ω_b(g_sfc, g_cols, b_cols, z_cols, Dxs, Dys, ε², f, b, showplots=false)
ωx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
ωy = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
χx = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
χy = [zeros(g_cols[k].np) for k=1:g_sfc.nt]
for k=1:g_sfc.nt
    n = 0
    for i=1:3
        ig = g_sfc.t[k, i]
        nz = size(z_cols[ig], 1)
        i_col = n+1:n+nz 
        ωx[k][i_col] = ωx_b[k][i_col] + Ux[k]*ωx_Ux[k][i_col]/Hfield[ig]^2 - Uy[k]*ωy_Ux[k][i_col]/Hfield[ig]^2
        ωy[k][i_col] = ωy_b[k][i_col] + Ux[k]*ωy_Ux[k][i_col]/Hfield[ig]^2 + Uy[k]*ωx_Ux[k][i_col]/Hfield[ig]^2
        χx[k][i_col] = χx_b[k][i_col] + Ux[k]*χx_Ux[k][i_col]/Hfield[ig]^2 - Uy[k]*χy_Ux[k][i_col]/Hfield[ig]^2
        χy[k][i_col] = χy_b[k][i_col] + Ux[k]*χy_Ux[k][i_col]/Hfield[ig]^2 + Uy[k]*χx_Ux[k][i_col]/Hfield[ig]^2
        n += nz
    end
end
plot_ω_χ(ωx, ωy, χx, χy, g_cols)

# dx = 0.04
# x = -1+dx:dx:1-dx
# nx = size(x, 1)
# nz = nx
# z = -(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
# HH = [H([x[i], 0]) for i=1:nx]
# xx = repeat(x, 1, nz)
# zz = repeat(z', nx, 1).*repeat(HH, 1, nz)
# bb = [b([xx[i, j], 0, zz[i, j]]) for i=1:nx, j=1:nz]

# ux = zeros(nx, nz)
# uy = zeros(nx, nz)
# @showprogress for i=1:nx, j=1:nz
#     pt = [xx[i, j], 0, zz[i, j]]
#     try
#         k_sfc = nuPGCM.get_k(pt[1:2], g_sfc)
#         k = nuPGCM.get_k(pt, g)
#         ux[i, j] = ωy_b(pt, k) - Ux[k_sfc]*ωy_Ux(pt, k)/HH[i]^2 + Uy[k_sfc]*ωx_Ux(pt, k)/HH[i]^2
#         uy[i, j] = ωx_b(pt, k) + Ux[k_sfc]*ωx_Ux(pt, k)/HH[i]^2 - Uy[k_sfc]*ωy_Ux(pt, k)/HH[i]^2
#     catch
#         continue
#     end
# end
# for i=1:nx
#     ux[i, :] = +cumtrapz(ux[i, :], zz[i, :])
#     uy[i, :] = -cumtrapz(uy[i, :], zz[i, :])
# end

# fig, ax = plt.subplots(1)
# img = ax.pcolormesh(xx, zz, ux, cmap="RdBu_r", rasterized=true, shading="auto", vmin=-0.03, vmax=0.03)
# cb = colorbar(img, ax=ax, label=L"Cross-slope flow $u^x$")
# levels = -1:0.05:0
# ax.contour(xx, zz, bb, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
# ax.fill_between(xx[:, 1], zz[:, 1], minimum(zz), color="k", alpha=0.3, lw=0.0)
# ax.set_xlabel(L"Zonal coordinate $x$")
# ax.set_ylabel(L"Vertical coordinate $z$")
# ax.spines["left"].set_visible(false)
# ax.spines["bottom"].set_visible(false)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 0)
# savefig("scratch/images/ux.png")
# println("scratch/images/ux.png")
# plt.close()

# fig, ax = plt.subplots(1)
# img = ax.pcolormesh(xx, zz, uy, cmap="RdBu_r", rasterized=true, shading="auto", vmin=-0.15, vmax=0.15)
# cb = colorbar(img, ax=ax, label=L"Along-slope flow $u^y$")
# levels = -1:0.05:0
# ax.contour(xx, zz, bb, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
# ax.fill_between(xx[:, 1], zz[:, 1], minimum(zz), color="k", alpha=0.3, lw=0.0)
# ax.set_xlabel(L"Zonal coordinate $x$")
# ax.set_ylabel(L"Vertical coordinate $z$")
# ax.spines["left"].set_visible(false)
# ax.spines["bottom"].set_visible(false)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 0)
# savefig("scratch/images/uy.png")
# println("scratch/images/uy.png")
# plt.close()

println("Done.")