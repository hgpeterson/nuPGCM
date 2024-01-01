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
    ε² = 1e-2
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

    # t_final = 10
    # t_save = 0.1
    # Δt = 1e-6
    # t_final = 1e-3
    # t_save = 1e-5
    # evolve!(m, s, t_final, t_save; Δt)
    return s
end

function postprocess()
    i = 100
    while isfile("$out_folder/state$i.h5")
        s = load_state_3D(m, "$out_folder/state$i.h5")
        title = latexstring(L"$t = $", @sprintf("%.3f", s.t[1]))
        filename = @sprintf("%s/psi%03d.png", out_folder, i)
        nuPGCM.quick_plot(s.Ψ,  cb_label=L"Barotropic streamfunction $\Psi$", title=title, filename=filename)
        nuPGCM.plot_u(m, s, 0, i=i)
        i += 100
    end
    # run(`bash -c "make_movie 20 psi"`)
end

# m = setup()
# m = load_setup_3D("$out_folder/setup.h5")
# m = load_setup_3D("/home/hppeters/ResearchCallies/group_dir/sim008/output/setup.h5")
# s = run3d(m)
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

println("Done.")