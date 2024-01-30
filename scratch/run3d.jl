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
    ε² = 1e-3
    μϱ = 1e-4
    f = 1.
    # β = 0.
    β = 0.95
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
# m = load_setup_3D("../../group_dir/sim011/adv_on/output/data/setup.h5")
m = load_setup_3D("../../group_dir/sim012/adv_on/output/data/setup.h5")
# s = load_state_3D(m, "$out_folder/data/state5.h5")
s = run3d(m)
# postprocess()

println("Done.")