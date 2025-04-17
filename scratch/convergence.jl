using nuPGCM
using Gridap
using LinearAlgebra
using JLD2
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

set_out_dir!(".")

function solve_problem(; dim, h, α)
    # architecture and dimension
    arch = CPU()

    # params/funcs
    ε = 2e-2
    params = Parameters(ε, α, 0., 0., 0.)
    f₀ = 1
    β = 0.0
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    ν(x) = 1
    force_build_inversion_matrices = true

    # mesh
    mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))
    # mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e_squash%d.msh", dim, h, 1, 1/α))
    n_dofs = length(mesh.dofs.p_inversion)
    @info @sprintf("h = %e, n_dofs = %e\n", h, n_dofs)

    # build inversion matrices
    A_inversion_fname = @sprintf("../matrices/A_inversion_%sD_%e_%e_%e_%e_%e.jld2", dim, h, ε, α, f₀, β)
    # A_inversion_fname = @sprintf("../matrices/A_inversion_%sD_%e_%e_%e_%e_%e_squash%d.jld2", dim, h, ε, α, f₀, β, 1/α)
    if force_build_inversion_matrices
        @warn "force_build_inversion_matrices is true, generating..."
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; A_inversion_ofile=A_inversion_fname)
    elseif !isfile(A_inversion_fname) 
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

    # # b = α⁻¹z (= N²z^* in dimensional coordinates)
    # set_b!(model, x -> x[3]/α)
    # b = exp
    set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))

    # invert
    invert!(model)

    return model
end

function compute_error(model)
    α = model.params.α
    u = model.state.u
    v = model.state.v
    w = model.state.w
    p = model.state.p
    dΩ = model.mesh.dΩ
    P = model.mesh.spaces.X_trial[4]
    umax = maximum(abs.(u.free_values))
    vmax = maximum(abs.(v.free_values))
    wmax = maximum(abs.(w.free_values))
    eu_L∞ = maximum([umax, vmax, wmax])
    eu_L2 = sqrt(sum( ∫( u*u + v*v + w*w )*dΩ ))
    eu_H1 = sqrt(sum( ∫( u*u + v*v + w*w + 
                         ∂x(u)*∂x(u) + ∂y(u)*∂y(u) + ∂z(u)*∂z(u) +
                         ∂x(v)*∂x(v) + ∂y(v)*∂y(v) + ∂z(v)*∂z(v) +
                         ∂x(w)*∂x(w) + ∂y(w)*∂y(w) + ∂z(w)*∂z(w) 
                       )*dΩ ))
    p = FEFunction(P, p.free_values.args[1]) # recompute p as a FEFunction to make sure it is zero-mean
    p0 = interpolate_everywhere(x->x[3]^2/2/α^2, P) # since P is a zero-mean space, Gridap will automatically subtract the mean
    ep_L2 = sqrt(sum( ∫( (p - p0)*(p - p0) )*dΩ ))
    @printf("         |u|_L∞ = %e\n", eu_L∞)
    @printf("         |u|_L2 = %e\n", eu_L2)
    @printf("         |u|_H1 = %e\n", eu_H1)
    @printf("         |p|_L2 = %e\n", ep_L2)
    @printf("|u|_H1 + |p|_L2 = %e\n", eu_H1 + ep_L2)

    return eu_L∞, eu_L2, eu_H1, ep_L2
end

function save_plots(model; h)
    α = model.params.α
    u = model.state.u
    v = model.state.v
    w = model.state.w
    P = model.mesh.spaces.X_trial[4]
    p = FEFunction(P, model.state.p.free_values.args[1])
    p0 = interpolate_everywhere(x->x[3]^2/2/α^2, P) 
    b = model.state.b
    plot_slice(u, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"u", fname=@sprintf("%s/images/u_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(v, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"v", fname=@sprintf("%s/images/v_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(w, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"w", fname=@sprintf("%s/images/w_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(u*u + v*v + w*w, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"$|\mathbf{u}|^2$", 
                fname=@sprintf("%s/images/u_L2_err_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(u*u + v*v + w*w + 
                ∂x(u)*∂x(u) + ∂y(u)*∂y(u) + ∂z(u)*∂z(u) +     
                ∂x(v)*∂x(v) + ∂y(v)*∂y(v) + ∂z(v)*∂z(v) +     
                ∂x(w)*∂x(w) + ∂y(w)*∂y(w) + ∂z(w)*∂z(w),
                b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"$|\mathbf{u}|^2 + |\nabla\mathbf{u}|^2$", 
                fname=@sprintf("%s/images/u_H1_err_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice((p - p0)*(p - p0), b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"$|p - p_a|^2$", 
                fname=@sprintf("%s/images/p_err_%1.2e_%1.2e.png", out_dir, h, α))
end

function plot_w(model; h)
    α = model.params.α
    w = model.state.w
    b = model.state.b
    plot_slice(w, b, 1/α; y=0, bbox=[-1, -α, 1, 0], fname=@sprintf("%s/images/w_%1.2e_%1.2e.png", out_dir, h, α))
end
function plot_profiles(model; h)
    α = model.params.α
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b
    plot_profiles(u, v, w, b, 1/α, x->α*(1 - x[1]^2); x=0.5, y=0, fname=@sprintf("%s/images/profiles_%1.2e_%1.2e.png", out_dir, h, α))
end

function compute_errors(; dim, hs, α)
    n_dofs = zeros(length(hs))
    eu_L∞ = zeros(length(hs))
    eu_L2 = zeros(length(hs))
    eu_H1 = zeros(length(hs))
    ep_L2 = zeros(length(hs))
    for i in eachindex(hs)
        model = solve_problem(dim=dim, h=hs[i], α=α)
        n_dofs[i], eu_L∞[i], eu_L2[i], eu_H1[i], ep_L2[i] = compute_error(model)
    end
    println()
    @printf("              h = [%1.5e, %1.5e, %1.5e]\n", hs...)
    @printf("         n_dofs = [%1.5e, %1.5e, %1.5e]\n", n_dofs...)
    @printf("         |u|_L∞ = [%1.5e, %1.5e, %1.5e]\n", eu_L∞...)
    @printf("         |u|_L2 = [%1.5e, %1.5e, %1.5e]\n", eu_L2...)
    @printf("         |u|_H2 = [%1.5e, %1.5e, %1.5e]\n", eu_H1...)
    @printf("         |p|_L2 = [%1.5e, %1.5e, %1.5e]\n", ep_L2...)
    @printf("|u|_H1 + |p|_L2 = [%1.5e, %1.5e, %1.5e]\n", (eu_H1 + ep_L2)...)
    return n_dofs, eu_L∞, eu_L2, eu_H1, ep_L2
end

function plot_convergence_2D()
    # n_dofs = [
    #     [4.02802e+05, 1.00333e+05, 2.52460e+04]
    # ]

    hs = [
        [1.00000e-02, 2.00000e-02, 5.00000e-02],
        [5.00000e-03, 1.00000e-02, 2.00000e-02]
    ]
    hmin = minimum(minimum(hs))
    hmax = maximum(maximum(hs))

    errors = [
        [1.15721e-04, 6.07887e-04, 5.90033e-03],
        [6.99864e-05, 3.03114e-04, 1.58356e-03]
    ]
    emin = minimum(minimum(errors))
    emax = maximum(maximum(errors))

    # eu_L∞ = [
    #     [1.37493e-06, 1.91026e-05, 4.91261e-05]
    # ]

    labels = [
        L"$\varepsilon = 10^{-1}$, $\alpha = 1/2$, aniso",
        L"$\varepsilon = 10^{-1}$, $\alpha = 1/2$, iso",
    ]

    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(true)
    ax.spines["right"].set_visible(true)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
    ax.set_xlim(0.9*hmin, 1.1*hmax)
    ax.set_ylim(0.9*emin, 1.1*emax)
    ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
    ax.set_axisbelow(true) # put grid behind lines
    for i ∈ eachindex(hs)
        ax.plot(hs[i], errors[i], "o-", label=labels[i])
    end
    ax.plot([hmin, hmax], 2emin/hmin^2*[hmin^2, hmax^2], "k--", label=L"$C h^2$")
    ax.legend(loc=(1.05, 0.0))
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

# dim = 2
# h = 2e-2
# α = 1/2
# model = solve_problem(; dim, h, α)
# n_dofs, eu_L2, eu_H1, ep_L2 = compute_error(model)
# save_plots(model; h)
# plot_profiles(model; h)
# plot_w(model; h)

# n_dofs, eu_L∞, eu_L2, eu_H1, ep_L2 = compute_errors(2, [5e-3, 1e-2, 2e-2])

plot_convergence_2D()
# plot_convergence_3D()

println("Done.")

### results for ε = 1e-1 

## squashed meshes (originating from H = 1 - x^2 mesh at uniform h = 1e-2 with n_dofs = 201152)

# α = 1:
#          |u|_L∞ = 8.992144e-07
#          |u|_L2 = 4.173351e-08
#          |u|_H1 = 2.526387e-05
#          |p|_L2 = 3.694708e-07
# |u|_H1 + |p|_L2 = 2.563334e-05

# α = 1/2:
#          |u|_L∞ = 2.716970e-06
#          |u|_L2 = 7.523147e-08
#          |u|_H1 = 7.185127e-05
#          |p|_L2 = 2.599289e-07
# |u|_H1 + |p|_L2 = 7.211120e-05 -> 2.88 times α = 1

# α = 1/4:
#          |u|_L∞ = 7.698220e-06
#          |u|_L2 = 1.362149e-07
#          |u|_H1 = 2.062875e-04
#          |p|_L2 = 1.826300e-07
# |u|_H1 + |p|_L2 = 2.064701e-04 -> 8.05 times α = 1

## isotropic meshes

# α = 1/2, h = 5e-3:
#          n_dofs = 4.028020e+05 -> double the dofs
#          |u|_L∞ = 1.374932e-06
#          |u|_L2 = 5.595901e-08
#          |u|_H1 = 6.970544e-05
#          |p|_L2 = 2.809806e-07
# |u|_H1 + |p|_L2 = 6.998642e-05 -> 0.97 times squashed mesh

# α = 1/2, h = 7e-3:
#          n_dofs = 2.050100e+05 -> same dofs
#          |u|_L∞ = 3.857573e-06
#          |u|_L2 = 1.247994e-07
#          |u|_H1 = 1.143373e-04
#          |p|_L2 = 4.446257e-07
# |u|_H1 + |p|_L2 = 1.147819e-04 -> 1.59 times squashed mesh ??

# α = 1/2, h = 1e-2:
#          n_dofs = 1.003330e+05 -> half the dofs
#          |u|_L∞ = 1.910264e-05
#          |u|_L2 = 4.901199e-07
#          |u|_H1 = 3.017957e-04
#          |p|_L2 = 1.318180e-06
# |u|_H1 + |p|_L2 = 3.031139e-04 -> 4.20 times squashed mesh