using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

if !isdir("../output")
    mkdir("../output")
end
set_out_folder("../output")
if !isdir("$out_folder/data")
    mkdir("$out_folder/data")
end
if !isdir("$out_folder/images")
    mkdir("$out_folder/images")
end

# depth
H(x) = 1 - x[1]^2 - x[2]^2

function setup()
    # params
    ε² = 1e-4
    μϱ = 1e-4
    f = 1.
    β = 0.
    # β = 0.95
    params = Params(; ε², μϱ, f, β)

    # geometry
    geom = Geometry(:circle, H, res=3, nσ=0, chebyshev=true)

    # forcing
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ(σ, H) = 1 + 0*σ*H
    ν(σ, H) = κ(σ, H)
    forcing = Forcing(geom, τx, τy, ν, κ)

    # setup and save
    m = ModelSetup3D(params, geom, forcing, advection=true)
    save_setup(m)

    return m
end

function run3d(m::ModelSetup3D)
    # b = FEField(x -> H(x)*x[3], m.geom.g2)
    b = FEField(x -> H(x)*x[3] + 0.1*exp(-(H(x)*x[3] + H(x))/0.1), m.geom.g2)
    s = initial_state(m, b)
    # s = initial_state(m, b, showplots=true)

    Δt = 1e-4
    t_save = 1e-3
    t_final = 1e-1
    # evolve!(m, s, t_final, t_save; Δt)
    return s
end

function postprocess()
    i = 1
    while isfile("$out_folder/data/state$i.h5")
        s = load_state_3D(m, "$out_folder/data/state$i.h5")
        title = latexstring(L"$t = $", @sprintf("%1.1e", s.t[1]))
        filename = @sprintf("%s/images/psi%03d.png", out_folder, i)
        nuPGCM.quick_plot(s.Ψ,  cb_label=L"Barotropic streamfunction $\Psi$", title=title, filename=filename)
        nuPGCM.plot_u(m, s, 0, i=i)
        i += 1
    end
    # run(`bash -c "make_movie 20 psi"`)
end

# m = setup()
# m = load_setup_3D("$out_folder/data/setup.h5")
# m = load_setup_3D("../../group_dir/sim012/adv_on/output/data/setup.h5")
# s = load_state_3D(m, "$out_folder/data/state5.h5")
# s = run3d(m)
# postprocess()

# m = load_setup_3D("../../group_dir/sim011/adv_on/output/data/setup.h5")
# s = load_state_3D(m, "../../group_dir/sim011/adv_on/output/data/state10.h5")

# using PyCall
# lines = pyimport("matplotlib.lines")
# Ux, Uy = nuPGCM.compute_U(s.Ψ)
# Ux = FEField(Ux)
# Uy = FEField(Uy)
# x = -1.0:0.01:1.0
# y = 0
# fig, ax = plt.subplots(1)
# ax.axhline(0, c="k", ls="--", lw=0.25)
# ax.set_xlim(-1, 1)
# ax.set_ylim(-0.06, 0.06)
# ax.set_xlabel(L"Zonal coordinate $x$")
# ax.set_ylabel("Transport")
# ax.plot(x, [Ux([x₀, y]) for x₀ ∈ x], "C0-")
# ax.plot(x, Ux_beta, "C0--")
# ax.plot(x, [Uy([x₀, y]) for x₀ ∈ x], "C1-")
# ax.plot(x, Uy_beta, "C1--")
# custom_handles = [lines.Line2D([0], [0], c="C0", ls="-",  lw=1),
#                   lines.Line2D([0], [0], c="C1", ls="-",  lw=1),
#                   lines.Line2D([0], [0], c="k",  ls="-",  lw=1),
#                   lines.Line2D([0], [0], c="k",  ls="--", lw=1)]
# custom_labels = [L"U^x", L"U^y", L"$f$-plane", L"$\beta$-plane"]
# ax.legend(custom_handles, custom_labels, ncol=2)
# savefig("$out_folder/images/U.png")
# println("$out_folder/images/U.png")

# # f/H
# f = m.params.f
# β = m.params.β
# g_sfc2 = m.geom.g_sfc2
# f_over_H = FEField(x->f + β*x[2], g_sfc2)/m.geom.H
# vmax = 1e1
# f_over_H.values[g_sfc2.e["bdy"]] .= vmax
# nuPGCM.quick_plot(f_over_H, cb_label=L"f/H", filename="$out_folder/images/f_over_H.png"; vmax, contour_levels=10)

# # Psi
# nuPGCM.quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", filename="$out_folder/images/psi.png")

# # ux, uy
# nuPGCM.plot_u(m, s, 0; title="")

# ωx_b, ωy_b, χx_b, χy_b, Ux_BL_b, Uy_BL_b = nuPGCM.solve_baroclinic_buoyancy_BL(m, s.b)

# g_sfc1 = m.geom.g_sfc1
# τx_b_bot = DGField(ωy_b[:, :, 1], g_sfc1)
# τy_b_bot = DGField(-ωx_b[:, :, 1], g_sfc1)
# τ_b_bot = √(τx_b_bot^2 + τy_b_bot^2)

# dr = 0.2
# r = dr:dr:1-dr
# dθ = π/12
# θ = 0:dθ:2π-dθ
# x = [rᵢ*cos(θⱼ) for rᵢ ∈ r, θⱼ ∈ θ][:]
# y = [rᵢ*sin(θⱼ) for rᵢ ∈ r, θⱼ ∈ θ][:]
# u = [τx_b_bot([x[i], y[i]]) for i ∈ eachindex(x)]
# v = [τy_b_bot([x[i], y[i]]) for i ∈ eachindex(x)]

# fig, ax, im = nuPGCM.tplot(FEField(τ_b_bot), cb_label=L"|\vec{\tau}^b(-H)|")
# ax.quiver(x, y, u, v)
# ax.set_xlabel(L"Zonal coordinate $x$")
# ax.set_ylabel(L"Meridional coordinate $y$")
# ax.axis("equal")
# ax.set_xticks(-1:0.5:1)
# ax.set_yticks(-1:0.5:1)
# savefig("$out_folder/images/tau_b_bot_BL_quiver.png")
# println("$out_folder/images/tau_b_bot_BL_quiver.png")
# plt.close()

println("Done.")
