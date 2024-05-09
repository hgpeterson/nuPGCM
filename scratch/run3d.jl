using nuPGCM
using PyPlot
using Printf
using SpecialFunctions

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
    μϱ = 1e0
    f = 1.
    β = 0.
    δ₀ = 1e-1
    params = Params(; ε², μϱ, f, β, δ₀)

    # geometry
    geom = Geometry(:circle, H, res=4, chebyshev=true)

    # forcing
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ(σ, H) = 1 + 0*σ*H
    # ν(σ, H) = κ(σ, H)
    ν(σ, H) = 1 + 0*σ*H
    forcing = Forcing(geom, τx, τy, ν, κ)

    # setup and save
    m = ModelSetup3D(params, geom, forcing, advection=true)
    save_setup(m)

    return m
end

function run3d(m::ModelSetup3D)
    b = FEField(x -> H(x)*x[3], m.geom.g2)
    # b = FEField(x -> H(x)*x[3] + 0.1*exp(-(H(x)*x[3] + H(x))/0.1), m.geom.g2)
    # b = FEField(x -> -cos(π*x[1]/2), m.geom.g2)
    # b = FEField(x -> -exp(-(x[1]^2 + x[2]^2 + (x[3]*H(x) + 0.5)^2)/0.02), m.geom.g2)
    s = initial_state(m, b; showplots=false)
    # nuPGCM.plot_u(m, s, 0, i=0)
    # nuPGCM.plot_profiles(m, s; x=0.25, y=0.0)

    Δt = 1e-4 
    t_plot = 1e-1
    t_save = 1e-1
    t_final = 10
    evolve!(m, s, t_final, t_plot, t_save; Δt)#, i_save=7, i_plot=70)
    return s
end

# m = setup()
# m = load_setup_3D("$out_folder/data/setup.h5")
# m = load_setup_3D("../../group_dir/sim017/d0.1/output/data/setup.h5")
# s = load_state_3D(m, "$out_folder/data/state7.h5")
s = run3d(m)

println("Done.")
