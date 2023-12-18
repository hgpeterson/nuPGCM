using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

# depth
H(x) = 1 - x[1]^2 - x[2]^2

function setup()
    # params
    μ = 1e4
    ε² = 1e-3
    ϱ = 1e-4
    # Δt = 1e-4*μ*ϱ/ε²
    Δt = 1e-2
    f = 1.
    β = 0.
    # β = 1.
    params = Params(; ε², μ, ϱ, Δt, f, β)

    # geometry
    geom = Geometry(:circle, H, res=4, nσ=0, chebyshev=true)

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
    ωx = DGField(0, m.geom.g1)
    ωy = DGField(0, m.geom.g1)
    χx = DGField(0, m.geom.g1)
    χy = DGField(0, m.geom.g1)
    Ψ = FEField(0, m.geom.g_sfc1)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, [0])

    invert!(m, s, showplots=true)

    # t_final = 10
    # t_plot = 1
    # t_save = 0.1
    # evolve!(m, s, t_final, t_plot, t_save)
    return s
end

function postprocess()
    i = 1
    while isfile("$out_folder/state$i.h5")
        s = load_state_3D(m, "$out_folder/state$i.h5")
        title = latexstring(L"$t = $", @sprintf("%.3f", m.params.Δt*s.i[1]))
        filename = @sprintf("%s/psi%03d.png", out_folder, i)
        nuPGCM.quick_plot(s.Ψ,  cb_label=L"Barotropic streamfunction $\Psi$", title=title, filename=filename)
        nuPGCM.plot_u(m, s, 0, i=i)
        i += 1
    end
    # run(`bash -c "make_movie 20 psi"`)
end

# m = setup()
m = load_setup_3D("$out_folder/setup4.h5")
s = run3d(m)
# postprocess()

# nuPGCM.plot_u(m, s, 0)

# for x ∈ 0.1:0.1:0.9
#     nuPGCM.plot_profiles(m, s, x=x, y=0.0, filename="$out_folder/profiles$x.png"; m2D, s2D)
# end
# for θ ∈ 0:π/4:2π
#     nuPGCM.plot_profiles(m, s, x=0.5*cos(θ), y=0.5*sin(θ), filename="$out_folder/profiles_th$θ.png"; m2D, s2D)
# end

# b_prod = nuPGCM.buoyancy_production(m, s) 
# ke_diss = nuPGCM.KE_dissipation(m, s)
# println("KE:")
# @printf("    ∫uᶻb   = % .5e\n", b_prod)
# @printf("    ε²∫νω² = % .5e\n", ke_diss)
# @printf("    error  = % .5e\n", abs(ke_diss - b_prod))

function test_energy(m::ModelSetup3D)
    b = FEField(x -> H(x)*x[3] + 0.1*exp(-(H(x)*x[3] + H(x))/0.1), m.geom.g2)
    ωx = DGField(0, m.geom.g1)
    ωy = DGField(0, m.geom.g1)
    χx = DGField(0, m.geom.g1)
    χy = DGField(0, m.geom.g1)
    Ψ = FEField(0, m.geom.g_sfc1)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, [0])
    invert!(m, s)
    # invert!(m, s, showplots=true)
    nuPGCM.plot_u(m, s, 0.0)

    b_prod = nuPGCM.buoyancy_production(m, s) 
    ke_diss = nuPGCM.KE_dissipation(m, s)
    println("KE:")
    @printf("    ∫uᶻb   = % .5e\n", b_prod)
    @printf("    ε²∫νω² = % .5e\n", ke_diss)
    @printf("    error  = % .5e\n", abs(ke_diss - b_prod))

    x = 0.5
    y = 0.0
    nσ = m.geom.nσ
    g_sfc1 = m.geom.g_sfc1
    g1 = m.geom.g1
    k_sfc = nuPGCM.get_k([x, y], g_sfc1, g_sfc1.el)
    ξ_sfc = transform_to_ref_el(g_sfc1.el, [x, y], g_sfc1.p[g_sfc1.t[k_sfc, :], :])

    σ = -1:0.01:0
    H₀ = m.geom.H(ξ_sfc, k_sfc)
    Hx₀ = m.geom.Hx(ξ_sfc, k_sfc)
    Hy₀ = m.geom.Hy(ξ_sfc, k_sfc)
    z = σ*H₀
    k_ws = [nuPGCM.get_k_w(k_sfc, nσ, findfirst(j -> m.geom.σ[j] ≤ σ[i] ≤ m.geom.σ[j+1], 1:nσ)) for i ∈ eachindex(σ)] 
    ξ_ws = [transform_to_ref_el(g1.el, [x, y, σ[i]], g1.p[g1.t[k_ws[i], :], :]) for i ∈ eachindex(σ)]

    ωx = [s.ωx(ξ_ws[i], k_ws[i]) for i ∈ eachindex(σ)]
    ωy = [s.ωy(ξ_ws[i], k_ws[i]) for i ∈ eachindex(σ)]
    ν = [m.forcing.ν(ξ_ws[i], k_ws[i]) for i ∈ eachindex(σ)]
    b = [s.b(ξ_ws[i], k_ws[i]) for i ∈ eachindex(σ)]
    ux = [-∂z(s.χy, ξ_ws[i], k_ws[i])/H₀ for i ∈ eachindex(σ)]
    uy = [+∂z(s.χx, ξ_ws[i], k_ws[i])/H₀ for i ∈ eachindex(σ)]
    Huσ = [∂x(s.χy, ξ_ws[i], k_ws[i]) - ∂y(s.χx, ξ_ws[i], k_ws[i]) for i ∈ eachindex(σ)]
    uz = @. Huσ + σ*Hx₀*ux + σ*Hy₀*uy
    ux_z = differentiate(ux, z)
    uy_z = differentiate(uy, z)
    ke_diss = @. m.params.ε²*ν*(ux_z^2 + uy_z^2)
    @printf("ke_diss = % .5e\n", trapz(ke_diss, z))
    ke_diss = @. m.params.ε²*ν*(ωx^2 + ωy^2)
    b_prod = @. b*uz
    @printf("ke_diss = % .5e\n", trapz(ke_diss, z))
    @printf("b_prod  = % .5e\n", trapz(b_prod, z))

    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.plot(ke_diss, z, label=L"\varepsilon^2 \nu \omega^2")
    ax.plot(b_prod, z, label=L"u^z b")
    ax.legend()
    savefig("$out_folder/debug.png")
    println("$out_folder/debug.png")
    plt.close()

    return s
end

# s = test_energy(m)

println("Done.")