using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    ε² = 1e-4
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 1.
    β = 1.
    H(x) = 1 - x[1]^2 - x[2]^2
    τx(x) = 0.
    τy(x) = 0.

    g_sfc1 = Grid(1, "meshes/circle/mesh2.h5")

    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H::Function, τx::Function, τy::Function, g_sfc1)

    σ = -1:0.01:0
    nσ = length(σ)
    H = 1e-10
    A = nuPGCM.get_baroclinic_LHS(σ*H, ε², f)
    r = nuPGCM.get_baroclinic_RHS(σ*H, zeros(nσ), zeros(nσ), H^2, 0, 0, 0, ε²)
    sol = A\r
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
    ax[1].plot(sol[0*nσ+1:1*nσ], σ*H)
    ax[1].plot(sol[1*nσ+1:2*nσ], σ*H)
    ax[2].plot(sol[2*nσ+1:3*nσ], σ*H)
    ax[2].plot(sol[3*nσ+1:4*nσ], σ*H)
    ax[1].set_xlabel(L"\omega")
    ax[2].set_xlabel(L"\chi")
    ax[1].set_ylabel(L"z")
    savefig("scratch/images/omega_chi.png")
    println("scratch/images/omega_chi.png")

    # m = ModelSetup3D()
    # # b = [FEField(x -> x[3], g) for g ∈ m.b_cols]
    # b = [FEField(x -> H(x)*x[3]^2 + 2/3*x[3]^3, g) for g ∈ m.b_cols]
    # # δ = 0.1
    # # b = [FEField(x -> x[3] + δ*exp(-(x[3] + H(x))/δ), g) for g ∈ m.b_cols]
    # ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    # s = ModelState3D(b, ωx, ωy, χx, χy, 0)
    # # evolve!(m, s)
    # return m, s
end

# m, s = bowl()

println("Done.")