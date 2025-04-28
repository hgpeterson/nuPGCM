using nuPGCM
using Gridap
using LinearAlgebra
using JLD2
using Printf
using PyPlot

import nuPGCM: plot_profiles

ENV["JULIA_DEBUG"] = nuPGCM

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

set_out_dir!(".")

function setup_model()
    # mesh
    mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))
    @info n_pts(mesh)

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
    return inversion_model(arch, params, mesh, inversion_toolkit)
end

function solve_flat_isopycnal_problem!(model)
    # b = α⁻¹z (= N²z^* in dimensional coordinates)
    set_b!(model, x -> x[3]/α)
    invert!(model)
    return model
end

function solve_exponential_problem!(model)
    # b ~ exp()
    set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))
    invert!(model)
    return model
end

function constructed_problem_rhs(model; u0, v0, w0, p0)
    X_test = model.mesh.spaces.X_test
    dΩ = model.mesh.dΩ
    α²ε² = α^2*ε^2
    u0 = interpolate_everywhere(u0, model.mesh.spaces.X_trial[1])
    v0 = interpolate_everywhere(v0, model.mesh.spaces.X_trial[2])
    w0 = interpolate_everywhere(w0, model.mesh.spaces.X_trial[3])
    p0 = interpolate_everywhere(p0, model.mesh.spaces.X_trial[4])
    l((vx, vy, vz, q)) =  
        ∫(α²ε²*∂x(u0)*∂x(vx)*ν + α²ε²*∂y(u0)*∂y(vx)*ν + α²ε²*∂z(u0)*∂z(vx)*ν - v0*vx*f + ∂x(p0)*vx +
          α²ε²*∂x(v0)*∂x(vy)*ν + α²ε²*∂y(v0)*∂y(vy)*ν + α²ε²*∂z(v0)*∂z(vy)*ν + u0*vy*f + ∂y(p0)*vy +
          α²ε²*∂x(w0)*∂x(vz)*ν + α²ε²*∂y(w0)*∂y(vz)*ν + α²ε²*∂z(w0)*∂z(vz)*ν +           ∂z(p0)*vz +
                                                                    ∂x(u0)*q + ∂y(v0)*q + ∂z(w0)*q )dΩ
    @time "rhs" rhs = assemble_vector(l, X_test)
    rhs = rhs[model.mesh.dofs.p_inversion]
    return rhs
end
function constructed_problem_rhs(model; f₁, f₂, f₃, f₄)
    X_test = model.mesh.spaces.X_test
    dΩ = model.mesh.dΩ
    l((vx, vy, vz, q)) = ∫( f₁*vx + f₂*vy + f₃*vz + f₄*q )dΩ
    @time "rhs" rhs = assemble_vector(l, X_test)
    rhs = rhs[model.mesh.dofs.p_inversion]
    return rhs
end

function solve_constructed_problem!(model, rhs)
    model.inversion.solver.y .= on_architecture(arch, rhs)
    nuPGCM.iterative_solve!(model.inversion.solver)
    nuPGCM.sync_flow!(model)
    return model
end

n_pts(mesh) = mesh.dofs.np + 1 # pressure lives on the nodes but it has one less dof due to the zero-mean constraint

function compute_error(model)
    # unpack
    P = model.mesh.spaces.X_trial[4]
    dΩ = model.mesh.dΩ

    # errors
    δu = model.state.u - u0
    δv = model.state.v - v0
    δw = model.state.w - w0
    δp = FEFunction(P, model.state.p.free_values.args[1]) - p0 # have to make sure p is zero-mean
    
    # compute integrals
    δu_H1 = sqrt(sum( ∫( δu*δu + δv*δv + δw*δw +
                         ∂x(δu)*∂x(δu) + ∂y(δu)*∂y(δu) + ∂z(δu)*∂z(δu) +
                         ∂x(δv)*∂x(δv) + ∂y(δv)*∂y(δv) + ∂z(δv)*∂z(δv) +
                         ∂x(δw)*∂x(δw) + ∂y(δw)*∂y(δw) + ∂z(δw)*∂z(δw)
                       )*dΩ ))
    δp_L2 = sqrt(sum( ∫( δp*δp )*dΩ ))
    @printf("|δu|_H1 + |δp|_L2 = %e + %e = %e\n", δu_H1, δp_L2, δu_H1 + δp_L2)

    return δu_H1 + δp_L2
end

function save_plots(model)
    δu = model.state.u - u0
    δv = model.state.v - v0
    δw = model.state.w - w0
    P = model.mesh.spaces.X_trial[4]
    δp = FEFunction(P, model.state.p.free_values.args[1]) - p0 # have to make sure p is zero-mean
    b = model.state.b
    plot_slice(δu, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"u - u_0", fname=@sprintf("%s/images/u_err_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(δv, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"v - v_0", fname=@sprintf("%s/images/v_err_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(δw, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"w - w_0", fname=@sprintf("%s/images/w_err_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(δu*δu + δv*δv + δw*δw +
               ∂x(δu)*∂x(δu) + ∂y(δu)*∂y(δu) + ∂z(δu)*∂z(δu) +
               ∂x(δv)*∂x(δv) + ∂y(δv)*∂y(δv) + ∂z(δv)*∂z(δv) +
               ∂x(δw)*∂x(δw) + ∂y(δw)*∂y(δw) + ∂z(δw)*∂z(δw),
               b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"$|\mathbf{u} - \mathbf{u}_0|^2 + |\nabla(\mathbf{u} - \mathbf{u}_0)|^2$", 
               fname=@sprintf("%s/images/u_H1_err_%1.2e_%1.2e.png", out_dir, h, α))
    plot_slice(δp*δp, b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"$|p - p_0|^2$", 
               fname=@sprintf("%s/images/p_err_%1.2e_%1.2e.png", out_dir, h, α))
end

# function plot_w(model; h)
#     α = model.params.α
#     w = model.state.w
#     b = model.state.b
#     plot_slice(w, b, 1/α; y=0, bbox=[-1, -α, 1, 0], fname=@sprintf("%s/images/w_%1.2e_%1.2e.png", out_dir, h, α))
# end
# function plot_profiles(model; h)
#     α = model.params.α
#     u = model.state.u
#     v = model.state.v
#     w = model.state.w
#     b = model.state.b
#     plot_profiles(u, v, w, b, 1/α, x->α*(1 - x[1]^2); x=0.5, y=0, fname=@sprintf("%s/images/profiles_%1.2e_%1.2e.png", out_dir, h, α))
# end

# function compute_errors(; dim, hs, α)
#     n_dofs = zeros(length(hs))
#     eu_L∞ = zeros(length(hs))
#     eu_L2 = zeros(length(hs))
#     eu_H1 = zeros(length(hs))
#     ep_L2 = zeros(length(hs))
#     for i in eachindex(hs)
#         model = solve_problem(dim=dim, h=hs[i], α=α)
#         n_dofs[i], eu_L∞[i], eu_L2[i], eu_H1[i], ep_L2[i] = compute_error(model)
#     end
#     println()
#     @printf("              h = [%1.5e, %1.5e, %1.5e]\n", hs...)
#     @printf("         n_dofs = [%1.5e, %1.5e, %1.5e]\n", n_dofs...)
#     @printf("         |u|_L∞ = [%1.5e, %1.5e, %1.5e]\n", eu_L∞...)
#     @printf("         |u|_L2 = [%1.5e, %1.5e, %1.5e]\n", eu_L2...)
#     @printf("         |u|_H2 = [%1.5e, %1.5e, %1.5e]\n", eu_H1...)
#     @printf("         |p|_L2 = [%1.5e, %1.5e, %1.5e]\n", ep_L2...)
#     @printf("|u|_H1 + |p|_L2 = [%1.5e, %1.5e, %1.5e]\n", (eu_H1 + ep_L2)...)
#     return n_dofs, eu_L∞, eu_L2, eu_H1, ep_L2
# end

# function plot_convergence_2D()
#     # n_dofs = [
#     #     [4.02802e+05, 1.00333e+05, 2.52460e+04]
#     # ]

#     hs = [
#         [1.00000e-02, 2.00000e-02, 5.00000e-02],
#         [5.00000e-03, 1.00000e-02, 2.00000e-02]
#     ]
#     hmin = minimum(minimum(hs))
#     hmax = maximum(maximum(hs))

#     errors = [
#         [1.15721e-04, 6.07887e-04, 5.90033e-03],
#         [6.99864e-05, 3.03114e-04, 1.58356e-03]
#     ]
#     emin = minimum(minimum(errors))
#     emax = maximum(maximum(errors))

#     # eu_L∞ = [
#     #     [1.37493e-06, 1.91026e-05, 4.91261e-05]
#     # ]

#     labels = [
#         L"$\varepsilon = 10^{-1}$, $\alpha = 1/2$, aniso",
#         L"$\varepsilon = 10^{-1}$, $\alpha = 1/2$, iso",
#     ]

#     fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.spines["top"].set_visible(true)
#     ax.spines["right"].set_visible(true)
#     ax.set_xlabel(L"Resolution $h$")
#     ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
#     ax.set_xlim(0.9*hmin, 1.1*hmax)
#     ax.set_ylim(0.9*emin, 1.1*emax)
#     ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
#     ax.set_axisbelow(true) # put grid behind lines
#     for i ∈ eachindex(hs)
#         ax.plot(hs[i], errors[i], "o-", label=labels[i])
#     end
#     ax.plot([hmin, hmax], 2emin/hmin^2*[hmin^2, hmax^2], "k--", label=L"$C h^2$")
#     ax.legend(loc=(1.05, 0.0))
#     savefig(@sprintf("%s/images/convergence2D.png", out_dir))
#     println(@sprintf("%s/images/convergence2D.png", out_dir))
#     plt.close()
# end

# function plot_convergence_3D()
#     hs = [9.62898e-03, 1.88410e-02, 4.45354e-02]

#     errs = [
#         [1.65234e-01, NaN, NaN],
#         [1.65596e-01, 6.42308e-01, 3.66494e+00],
#         [1.701087e-01, NaN, NaN],
#         [2.101807e-01, NaN, NaN],
#         [2.438967e-03, 2.211299e-02, 4.085575e-02],
#         [NaN, 6.901556e-03, 4.039652e-02],
#     ]
#     labels = [
#         L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
#         L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
#         L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-4}$",
#         L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-3}$",
#         L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-4}$",
#         L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
#     ]

#     println((errs[2][1] - errs[1][1])/errs[1][1])
#     println((errs[3][1] - errs[1][1])/errs[1][1])
#     println((errs[4][1] - errs[1][1])/errs[1][1])

#     fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.spines["top"].set_visible(true)
#     ax.spines["right"].set_visible(true)
#     ax.set_xlabel(L"Resolution $h$")
#     ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
#     ax.set_xlim(7e-3, 5e-2)
#     ax.set_ylim(1e-4, 1e1)
#     ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
#     ax.set_axisbelow(true)
#     for i ∈ eachindex(errs)
#         ax.plot(hs, errs[i], "o-", label=labels[i])
#     end
#     ax.plot(hs, errs[2][2]/hs[2]^2*hs.^2, "k--", label=L"$O(h^2)$")
#     ax.legend(loc=(1.05, 0.0))
#     ax.set_title("3D Bowl (Gridap)")
#     savefig(@sprintf("%s/images/convergence3D.png", out_dir))
#     println(@sprintf("%s/images/convergence3D.png", out_dir))
#     plt.close()
# end

function convergence_plot()
    Ns = [
        [549, 2061, 7955, 31458],
        [549, 2061, 7955, 31458]
    ]
    ds = [2, 2]
    Es = [
        [2.183774e-01, 2.407354e-02, 6.073092e-03, 2.122945e-03],
        [4.666244e-01, 7.929378e-02, 1.961613e-02, 6.420322e-03]
    ]
    labels = [
        L"2D, $\varepsilon = 10^{-2}$, $\alpha = 1/2$",
        L"2D, $\varepsilon = 10^{-2}$, $\alpha = 1/2$, $f_i$'s"
    ]

    pc = 1/6 
    fig, ax = plt.subplots(1, figsize=(19pc, 19pc))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"Resolution $N^{-1/d}$")
    ax.set_ylabel(L"Error $||\mathbf{u} - \mathbf{u}_0||_{H^1} + ||p - p_0||_{L^2}$")
    ax.set_xlim(1e-3, 1e-1)
    ax.set_ylim(1e-3, 1e0)
    for i ∈ eachindex(labels)
        ax.plot(Ns[i].^(-1/ds[i]), Es[i], "o-", label=labels[i])
    end
    h1, h2 = 1e-2, 4e-2
    ax.plot([h1, h2], 2e-3/h1^2*[h1^2, h2^2], "k-")
    ax.text(x=h2/2, y=1e-3/h1^2*(h2/2)^2, s=L"$h^2$")
    ax.legend(loc=(1.05, 0.0))
    savefig(@sprintf("%s/images/convergence.png", out_dir))
    println(@sprintf("%s/images/convergence.png", out_dir))
    # savefig(@sprintf("%s/images/convergence.pdf", out_dir))
    # println(@sprintf("%s/images/convergence.pdf", out_dir))
    plt.close()
end

# params/funcs
arch = CPU()
dim = 2
h = 1e-2
ε = 1e-1
α = 1/2
params = Parameters(ε, α, 0., 0., 0.)
f₀ = 1
β = 0.0
f(x) = f₀ + β*x[2]
H(x) = α*(1 - x[1]^2 - x[2]^2)
ν(x) = 1
force_build_inversion_matrices = false

# constructed solution
# u0(x) = 0
# v0(x) = 0
# w0(x) = 0
# p0 = interpolate_everywhere(x->x[3]^2/2/α^2, P) # since P is a zero-mean space, Gridap will automatically subtract the mean
u0(x) = sin(x[3] + H(x)) - cos(H(x))*(x[3] + H(x)) # must have u = 0 at z = -H and ∂z(u) = 0 at z = 0
v0(x) = sin(x[3] + H(x)) - cos(H(x))*(x[3] + H(x)) # must have v = 0 at z = -H and ∂z(v) = 0 at z = 0
w0(x) = sin(x[3] + H(x))*sin(x[3]) # must have w = 0 at z = -H, 0
# p0(x) = x[3]^2 - α^2/6 # 3D
p0(x) = x[3]^2 - 8α^2/35 # 2D

# -f*v + ∂x(p) - α²ε²∇⋅(ν∇u) = f₁
f₁(x) = f(x) * (x[3] - (-1 + x[1]^2 + x[2]^2) * α) * cos((-1 + x[1]^2 + x[2]^2) * α) -
    f(x) * sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α) - α^2 * ε^2 * (
    -4 * α * (-1 + x[1]^4 * α^2 + x[2]^4 * α^2 - x[2]^2 * α * (x[3] + α) -
    x[1]^2 * α * (x[3] + α - 2 * x[2]^2 * α)) * cos((-1 + x[1]^2 + x[2]^2) * α) -
    4 * α * cos(x[3] - (-1 + x[1]^2 + x[2]^2) * α) +
    4 * x[3] * α * sin((-1 + x[1]^2 + x[2]^2) * α) +
    4 * α^2 * sin((-1 + x[1]^2 + x[2]^2) * α) -
    12 * x[1]^2 * α^2 * sin((-1 + x[1]^2 + x[2]^2) * α) -
    12 * x[2]^2 * α^2 * sin((-1 + x[1]^2 + x[2]^2) * α) -
    sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α) -
    4 * x[1]^2 * α^2 * sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α) -
    4 * x[2]^2 * α^2 * sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α))
# f*u + ∂y(p) - α²ε²∇⋅(ν∇v) = f₂
f₂(x) = f(x) * (-((x[3] - (-1 + x[1]^2 + x[2]^2) * α) * cos((-1 + x[1]^2 + x[2]^2) * α)) +
    sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α)) - α^2 * ε^2 * (
    -4 * α * (-1 + x[1]^4 * α^2 + x[2]^4 * α^2 - x[2]^2 * α * (x[3] + α) -
    x[1]^2 * α * (x[3] + α - 2 * x[2]^2 * α)) * cos((-1 + x[1]^2 + x[2]^2) * α) -
    4 * α * cos(x[3] - (-1 + x[1]^2 + x[2]^2) * α) +
    4 * x[3] * α * sin((-1 + x[1]^2 + x[2]^2) * α) +
    4 * α^2 * sin((-1 + x[1]^2 + x[2]^2) * α) -
    12 * x[1]^2 * α^2 * sin((-1 + x[1]^2 + x[2]^2) * α) -
    12 * x[2]^2 * α^2 * sin((-1 + x[1]^2 + x[2]^2) * α) -
    sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α) -
    4 * x[1]^2 * α^2 * sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α) -
    4 * x[2]^2 * α^2 * sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α))
# ∂z(p) - α⁴ε²∇⋅(ν∇w) = f₃
f₃(x) = 2 * x[3] - α^4 * ε^2 * (
    2 * cos(x[3]) * cos(x[3] + (1 - x[1]^2 - x[2]^2) * α) -
    4 * α * cos(x[3] + (1 - x[1]^2 - x[2]^2) * α) * sin(x[3]) -
    2 * sin(x[3]) * sin(x[3] + (1 - x[1]^2 - x[2]^2) * α) -
    4 * x[1]^2 * α^2 * sin(x[3]) * sin(x[3] + (1 - x[1]^2 - x[2]^2) * α) -
    4 * x[2]^2 * α^2 * sin(x[3]) * sin(x[3] + (1 - x[1]^2 - x[2]^2) * α))
# ∂x(u) + ∂y(v) + ∂z(w) = f₄
f₄(x) = 2 * x[1] * α * cos((-1 + x[1]^2 + x[2]^2) * α) +
    2 * x[2] * α * cos((-1 + x[1]^2 + x[2]^2) * α) -
    2 * x[1] * α * cos(x[3] - (-1 + x[1]^2 + x[2]^2) * α) -
    2 * x[2] * α * cos(x[3] - (-1 + x[1]^2 + x[2]^2) * α) +
    cos(x[3] - (-1 + x[1]^2 + x[2]^2) * α) * sin(x[3]) -
    2 * x[1] * α * (-x[3] + (-1 + x[1]^2 + x[2]^2) * α) * sin((-1 + x[1]^2 + x[2]^2) * α) -
    2 * x[2] * α * (-x[3] + (-1 + x[1]^2 + x[2]^2) * α) * sin((-1 + x[1]^2 + x[2]^2) * α) +
    cos(x[3]) * sin(x[3] - (-1 + x[1]^2 + x[2]^2) * α)

# model = setup_model()
# # rhs = constructed_problem_rhs(model; u0, v0, w0, p0)
# rhs = constructed_problem_rhs(model; f₁, f₂, f₃, f₄)
# solve_constructed_problem!(model, rhs)
# compute_error(model)
# save_plots(model)

u0_fe = interpolate_everywhere(u0, model.mesh.spaces.X_trial[1])
v0_fe = interpolate_everywhere(v0, model.mesh.spaces.X_trial[2])
w0_fe = interpolate_everywhere(w0, model.mesh.spaces.X_trial[3])

plot_slice(u0_fe, model.state.b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"u", fname=@sprintf("%s/images/u0_%1.2e_%1.2e.png", out_dir, h, α))
plot_slice(v0_fe, model.state.b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"v", fname=@sprintf("%s/images/v0_%1.2e_%1.2e.png", out_dir, h, α))
plot_slice(w0_fe, model.state.b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"w", fname=@sprintf("%s/images/w0_%1.2e_%1.2e.png", out_dir, h, α))
plot_slice(model.state.u, model.state.b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"u", fname=@sprintf("%s/images/u_%1.2e_%1.2e.png", out_dir, h, α))
plot_slice(model.state.v, model.state.b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"v", fname=@sprintf("%s/images/v_%1.2e_%1.2e.png", out_dir, h, α))
plot_slice(model.state.w, model.state.b, 0; y=0, bbox=[-1, -α, 1, 0], cb_label=L"w", fname=@sprintf("%s/images/w_%1.2e_%1.2e.png", out_dir, h, α))

# convergence_plot()

println("Done.")

# 549:
# |δu|_H1 + |δp|_L2 = 4.608457e-01 + 5.778689e-03 = 4.666244e-01
# 2061:
# |δu|_H1 + |δp|_L2 = 7.896423e-02 + 3.295511e-04 = 7.929378e-02
# 7955
# |δu|_H1 + |δp|_L2 = 1.931750e-02 + 2.986271e-04 = 1.961613e-02
# 31458
# |δu|_H1 + |δp|_L2 = 6.287026e-03 + 1.332958e-04 = 6.420322e-03

# 2061:
# |δu|_H1 + |δp|_L2 = 1.965860e-02 + 1.678712e-03 = 2.133731e-02
# 7955:
# |δu|_H1 + |δp|_L2 = 2.025009e-02 + 1.728360e-03 = 2.197845e-02