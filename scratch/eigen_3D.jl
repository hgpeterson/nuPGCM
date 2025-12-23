using nuPGCM
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using Gridap, GridapGmsh
using KrylovKit, LinearOperators, LinearAlgebra, Krylov
using JLD2, Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

out_folder = "../out"

# architecture
arch = GPU()
tol = 1e-6
itmax = 0
VT = typeof(arch) == CPU ? Vector{ComplexF64} : CuVector{ComplexF64}

# params
hres = 0.05
ε² = 1e-2
γ = 1
f₀ = 1
β = 0.5
f(x) = f₀ + β*x[2]
H(x) = 1 - x[1]^2 - x[2]^2
ν(x) = 1
κ(x) = 1
μϱ = 1e0
dim = ThreeD()

function load_model()
    # model
    model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_%0.2f.msh", dim, hres))

    # triangulation and integration measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)

    # spaces for inversion
    X, Y, B, D = setup_FESpaces(model)
    Ux, Uy, Uz, P = unpack_spaces(X)
    nx = Ux.space.nfree
    ny = Uy.space.nfree
    nz = Uz.space.nfree
    nu = nx + ny + nz
    np = P.space.space.nfree
    nb = B.space.nfree
    N = nu + np - 1
    @printf("\nN = %d (%d + %d) ∼ 10^%d DOF\n", N, nu, np-1, floor(log10(N)))

    # filename for LHS matrix
    LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, hres, ε², γ, f₀, β)

    # inversion LHS
    if isfile(LHS_inversion_fname)
        LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
    else
        LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
    end

    # inversion RHS
    RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

    # preconditioner
    P_inversion = Diagonal(on_architecture(arch, 1/hres^3*ones(ComplexF64, N)))
    ldiv_P_inversion = false

    # put on GPU, if needed
    LHS_inversion = on_architecture(arch, ComplexF64.(LHS_inversion))
    RHS_inversion = on_architecture(arch, ComplexF64.(RHS_inversion))

    # # Krylov solver for inversion
    # solver = GmresSolver(N, N, 20, VT)
    # solver.x .= 0.

    # DOF
    nb = B.space.nfree
    println("DOF: ", nb)

    return LHS_inversion, RHS_inversion, P_inversion, X, Y, B, D, Ω, dΩ, nx, ny, nz, np, nb
end

# inversion function w = L(b)
function L(sol, LHS_inversion, RHS_inversion, b)
    b_arch = on_architecture(arch, b)
    if typeof(arch) == GPU
        RHS = [CUDA.zeros(nx); CUDA.zeros(ny); RHS_inversion*b_arch; CUDA.zeros(np-1)]
    else
        RHS = [zeros(nx); zeros(ny); RHS_inversion*b_arch; zeros(np-1)]
    end
    # Krylov.solve!(solver, LHS_inversion, RHS, solver.x, M=P_inversion, ldiv=ldiv_P_inversion,
    #             atol=tol, rtol=tol, verbose=0, itmax=itmax, restart=true,
    #             history=true)
    sol, stats = Krylov.gmres(LHS_inversion, RHS, sol, M=P_inversion,
                atol=tol, rtol=tol, verbose=0, itmax=itmax, restart=true, memory=20,
                history=true)
    @printf("inversion GMRES solve: solved=%s, niter=%d, time=%f\n", stats.solved, stats.niter, stats.timer)
    # @printf("inversion GMRES solve: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
    # sol = on_architecture(CPU(), solver.x[inv_perm_inversion])
    # sol = on_architecture(CPU(), solver.x)
    sol_cpu = on_architecture(CPU(), sol)
    return sol_cpu[nx+ny+1:nx+ny+nz]
end

# assemble
function assemble_system()
    # assemble K
    a_K(b, d) = ∫( ∂x(b)*∂x(d)*γ*κ + ∂y(b)*∂y(d)*γ*κ + ∂z(b)*∂z(d)*κ )dΩ
    @time "assemble K" K = assemble_matrix(a_K, B, D)

    # assemble M
    a_M(b, d) = ∫( b*d )dΩ
    @time "assemble M" M = assemble_matrix(a_M, B, D)

    # assemble M_w and M_wb
    _, _, Uz, _ = unpack_spaces(X)
    _, _, Vz, _ = unpack_spaces(Y)
    @time "assemble M_w" M_w = assemble_matrix(a_M, Uz, Vz)
    @time "assemble M_wb" M_wb = assemble_matrix(a_M, Uz, D)

    # Cuthill-McKee DOF reordering
    perm = nuPGCM.RCM_perm(arch, M)
    inv_perm = invperm(perm)
    # K = K[perm, perm]
    # M = M[perm, perm]
    # perm_w = nuPGCM.RCM_perm(arch, M_w)
    # M_wb = M_wb[perm, perm_w]

    return K, M, M_wb, inv_perm
end

# # load model
# # L, X, Y, B, D, Ω, dΩ, nb = load_model()
# LHS_inversion, RHS_inversion, P_inversion, X, Y, B, D, Ω, dΩ, nx, ny, nz, np, nb = load_model()

# # debug
# b = interpolate_everywhere(x->0.1*exp(-(x[3] + H(x))/0.1), B)
# _, _, Uz, _ = unpack_spaces(X)
# w = FEFunction(Uz, L(b.free_values))
# plot_slice(real(w), real(b), 1; x=0.0, cb_label=L"w", fname=@sprintf("%s/images/w_xslice.png", out_folder))

# # assemble system
# K, M, M_wb, inv_perm = assemble_system()
# @time "lu(M)" M = lu(M)

# solve B^-1 A X = ω X where A = -i*ε²/μϱ*K - i*M_wb*L and B = M
sol = CUDA.zeros(ComplexF64, nx+ny+nz+np-1)
@time vals, vecs, info = KrylovKit.eigsolve(x->M\(-im*ε²/μϱ*K*x - im*M_wb*L(sol, LHS_inversion, RHS_inversion, x)), nb, 1, :LR, ComplexF64, verbosity=3)
fname = @sprintf("%s/data/eigs.jld2", out_folder)
jldsave(fname; vals, vecs)
println(fname)

i = 1
ω = vals[i]
if imag(ω) >= 0
    @printf("ω = %e + %e i\n", real(ω), imag(ω))
else
    @printf("ω = %e - %e i\n", real(ω), -imag(ω))
end
# b = vecs[i][inv_perm]
b = vecs[i]
b /= 20*maximum(abs.(b)) # scale for plotting
b = FEFunction(B, b)
nuPGCM.plot_slice_wave(b, b, 1, k, ω; x=+0.0, cb_label=L"Re $b$", fname=@sprintf("%s/images/b_xslice.png", out_folder))
nuPGCM.plot_slice_wave(b, b, 1, k, ω; y=+0.0, cb_label=L"Re $b$", fname=@sprintf("%s/images/b_yslice.png", out_folder))
nuPGCM.plot_slice_wave(b, b, 1, k, ω; z=-0.5, cb_label=L"Re $b$", fname=@sprintf("%s/images/b_zslice.png", out_folder))

println("Done.")