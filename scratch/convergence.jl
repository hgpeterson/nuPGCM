using nuPGCM
using Statistics
using Gridap, GridapGmsh
using IncompleteLU, Krylov, LinearOperators, CuthillMcKee
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

set_out_dir!(".")

function compute_error(dim, h; showplots=false)
    # architecture and dimension
    arch = CPU()

    # params/funcs
    ε = 2e-2
    α = 1/2
    params = Parameters(ε, α, 0., 0., 0.)
    f₀ = 1
    β = 0.0
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    ν(x) = 1
    force_build_inversion_matrices = false

    # mesh
    mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))

    # build inversion matrices
    A_inversion_fname = @sprintf("../matrices/A_inversion_%sD_%e_%e_%e_%e_%e.jld2", dim, h, ε, α, f₀, β)
    if !isfile(A_inversion_fname) || force_build_inversion_matrices
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; A_inversion_ofile=A_inversion_fname)
    else
        file = jldopen(A_inversion_fname, "r")
        A_inversion = file["A_inversion"]
        close(file)
        B_inversion = nuPGCM.build_B_inversion(mesh, params)
    end

    # re-order dofs
    A_inversion = A_inversion[mesh.dofs.p_inversion, mesh.dofs.p_inversion]
    B_inversion = B_inversion[mesh.dofs.p_inversion, :]

    # preconditioner
    if typeof(arch) == CPU
        @time "lu(A_inversion)" P_inversion = lu(A_inversion)
    else
        P_inversion = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A_inversion, 1))))
    end

    # move to arch
    A_inversion = on_architecture(arch, A_inversion)
    B_inversion = on_architecture(arch, B_inversion)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion)

    # make an inverison model
    model = inversion_model(arch, params, mesh, inversion_toolkit)

    # b = z
    set_b!(model, x -> x[3])

    # invert
    invert!(model)

    # compute error
    u = model.state.u
    v = model.state.v
    w = model.state.w
    p = model.state.p
    dΩ = model.mesh.dΩ
    P = model.mesh.spaces.X_trial[4]
    eu_L2 = sqrt(sum( ∫( u*u + v*v + w*w )*dΩ ))
    eu_H1 = sqrt(sum( ∫( u*u + v*v + w*w + 
                         ∂x(u)*∂x(u) + ∂y(u)*∂y(u) + ∂z(u)*∂z(u) +
                         ∂x(v)*∂x(v) + ∂y(v)*∂y(v) + ∂z(v)*∂z(v) +
                         ∂x(w)*∂x(w) + ∂y(w)*∂y(w) + ∂z(w)*∂z(w) 
                       )*dΩ ))
    p0 = interpolate_everywhere(x->x[3]^2/2/α, P) # since P is a zero-mean space, Gridap will automatically subtract the mean
    ep_L2 = sqrt(sum( ∫( (p - p0)*(p - p0) )*dΩ ))
    @printf("    h = %e\n", h)
    @printf(" |u|₂ = %e\n", eu_L2)
    @printf(" |u|₁ = %e\n", eu_H1)
    @printf(" |p|₀ = %e\n", ep_L2)
    @printf("error = %e\n", eu_H1 + ep_L2)

    if showplots
        set_b!(model, x -> 0)
        b = model.state.b
        plot_slice(u*u + v*v + w*w, 
                   b, 1; y=0, cb_label=L"$|\mathbf{u}|^2$", 
                   fname=@sprintf("%s/images/u_L2_err_%1.2f.png", out_dir, h))
        plot_slice(u*u + v*v + w*w + 
                   ∂x(u)*∂x(u) + ∂y(u)*∂y(u) + ∂z(u)*∂z(u) +     
                   ∂x(v)*∂x(v) + ∂y(v)*∂y(v) + ∂z(v)*∂z(v) +     
                   ∂x(w)*∂x(w) + ∂y(w)*∂y(w) + ∂z(w)*∂z(w),
                   b, 1; y=0, cb_label=L"$|\mathbf{u}|^2 + |\nabla\mathbf{u}|^2$", 
                   fname=@sprintf("%s/images/u_H1_err_%1.2f.png", out_dir, h))
        plot_slice((p - p0)*(p - p0), 
                   b, 1; y=0, cb_label=L"$|p - p_a|^2$", fname=@sprintf("%s/images/p_err_%1.2f.png", out_dir, h))
    end

    return h, eu_L2, eu_H1, ep_L2
end

function plot_convergence_2D()
    # hs = [0.01, 0.02, 0.05]
    # hs = [7.96667e-03, 1.57642e-02, 3.80418e-02]
    hs = [9.96820e-03, 1.98866e-02, 4.91265e-02]

    errs = [
        [6.22111e-07, 3.42784e-06, 3.35299e-05], 

        [6.22111e-07, 3.42785e-06, 3.35303e-05], 

        [1.52113e-06, 8.14764e-06, 7.94241e-05], 
        
        [1.15721e-04, 6.07887e-04, 5.90033e-03],
        [1.50023e-04, 6.10646e-04, 5.90069e-03],
        [9.52791e-04, 7.61190e-04, 5.93063e-03],
        # [3.35898e-04, 6.14795e-04, 5.90047e-03],
        # [1.19563e-04, 6.14795e-04, 5.90047e-03],

        [1.15283e-02, 6.01363e-02, 5.67232e-01], 
        [1.16039e-02, 6.27308e-02, 5.78880e-01],
        [1.50846e-02, 6.27308e-02, 5.78880e-01],
        # [1.16003e-02, 6.27349e-02, 5.78861e-01],
        # [1.48924e-02, 6.27349e-02, 5.78861e-01]
    ]
    labels = [
        L"\varepsilon^2 = 10^{0},  \; f = 0, \; \gamma = 1",

        L"\varepsilon^2 = 10^{0},  \; f = 1, \; \gamma = 1",

        L"\varepsilon^2 = 10^{0},  \; f = 1, \; \gamma = 1/4",

        L"\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
        # L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-8}$",
        # L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-9}$",

        L"\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
    ]

    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(true)
    ax.spines["right"].set_visible(true)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
    ax.set_xlim(9e-3, 6e-2)
    ax.set_ylim(1e-7, 1e0)
    ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
    ax.set_axisbelow(true)
    for i ∈ eachindex(errs)
        ax.plot(hs, errs[i], "o-", label=labels[i])
    end
    ax.plot(hs, errs[1][2]/hs[2]^2*hs.^2, "k--", label=L"$O(h^2)$")
    ax.plot(hs, errs[4][2]/hs[2]^2*hs.^2, "k--")
    ax.plot(hs, errs[7][2]/hs[2]^2*hs.^2, "k--")
    hs = [1.00810e-02, 2.04083e-02, 5.16216e-02]
    ax.plot(hs, [7.91502e-03, 4.37259e-02, 4.04744e-01], "o-")
    ax.plot(hs, [7.94755e-05, 4.43226e-04, 4.27666e-03], "o-")
    ax.plot(hs, [7.17254e-04, 5.77746e-04, 4.31530e-03], "o-")
    ax.legend(loc=(1.05, 0.0))
    ax.set_title("2D Bowl (Gridap)")
    savefig(@sprintf("%s/images/convergence2D.png", out_dir))
    println(@sprintf("%s/images/convergence2D.png", out_dir))
    plt.close()
end

function plot_convergence_3D()
    hs = [9.62898e-03, 1.88410e-02, 4.45354e-02]

    errs = [
        [1.65234e-01, NaN, NaN],
        [1.65596e-01, 6.42308e-01, 3.66494e+00],
        [1.701087e-01, NaN, NaN],
        [2.101807e-01, NaN, NaN],
        [2.438967e-03, 2.211299e-02, 4.085575e-02],
        [NaN, 6.901556e-03, 4.039652e-02],
    ]
    labels = [
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-4}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-3}$",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-4}$",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
    ]

    println((errs[2][1] - errs[1][1])/errs[1][1])
    println((errs[3][1] - errs[1][1])/errs[1][1])
    println((errs[4][1] - errs[1][1])/errs[1][1])

    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(true)
    ax.spines["right"].set_visible(true)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
    ax.set_xlim(7e-3, 5e-2)
    ax.set_ylim(1e-4, 1e1)
    ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
    ax.set_axisbelow(true)
    for i ∈ eachindex(errs)
        ax.plot(hs, errs[i], "o-", label=labels[i])
    end
    ax.plot(hs, errs[2][2]/hs[2]^2*hs.^2, "k--", label=L"$O(h^2)$")
    ax.legend(loc=(1.05, 0.0))
    ax.set_title("3D Bowl (Gridap)")
    savefig(@sprintf("%s/images/convergence3D.png", out_dir))
    println(@sprintf("%s/images/convergence3D.png", out_dir))
    plt.close()
end

# showplots = false
showplots = true

dim = 2
# dim = 3

# h5, eu5_L2, eu5_H1, ep5_L2 = compute_error(dim, 0.05; showplots)
# h2, eu2_L2, eu2_H1, ep2_L2 = compute_error(dim, 0.02; showplots)
h1, eu1_L2, eu1_H1, ep1_L2 = compute_error(dim, 7e-3; showplots)

# @printf("[%1.5e, %1.5e, %1.5e]\n", h1, h2, h5)
# @printf("[%1.5e, %1.5e, %1.5e]\n", eu1_L2, eu2_L2, eu5_L2)
# @printf("[%1.5e, %1.5e, %1.5e]\n", eu1_H1, eu2_H1, eu5_H1)
# @printf("[%1.5e, %1.5e, %1.5e]\n", ep1_L2, ep2_L2, ep5_L2)
# @printf("[%1.5e, %1.5e, %1.5e]\n", eu1_H1 + ep1_L2, eu2_H1 + ep2_L2, eu5_H1 + ep5_L2)

# plot_convergence_2D()
# plot_convergence_3D()