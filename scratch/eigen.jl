using nuPGCM
using Gridap, GridapGmsh
using KrylovKit, LinearOperators, LinearAlgebra
using JLD2, Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
pc = 1/6
plt.close("all")

out_folder = "../out"

# architecture
arch = CPU()

# params
hres = 0.04
ε² = 1e-4
γ = 1/4
f₀ = 1
β = 1.0
f(x) = f₀ + β*x[2]
H(x) = 1 - x[1]^2 - x[2]^2
ν(x) = 1
κ(x) = 1
# κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)
μϱ = 1e0
k = -pi
dim = TwoD()

function load_model()
    # model
    model = GmshDiscreteModel(@sprintf("../meshes/bowl%sy_%0.2f.msh", dim, hres))

    # triangulation and integration measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)

    # reference FE 
    reffe_ux = ReferenceFE(lagrangian, Float64, 2; space=:P)
    reffe_uy = ReferenceFE(lagrangian, Float64, 2; space=:P)
    reffe_uz = ReferenceFE(lagrangian, Float64, 2; space=:P)
    reffe_p  = ReferenceFE(lagrangian, Float64, 1; space=:P)
    reffe_b  = ReferenceFE(lagrangian, Float64, 2; space=:P)

    # test FESpaces
    Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot"],        vector_type=Vector{ComplexF64})
    Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot"],        vector_type=Vector{ComplexF64})
    Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"], vector_type=Vector{ComplexF64})
    Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
    D  = TestFESpace(model, reffe_b,  conformity=:H1, dirichlet_tags=["sfc"],        vector_type=Vector{ComplexF64})
    Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

    # trial FESpaces with Dirichlet values
    Ux = TrialFESpace(Vx, [0])
    Uy = TrialFESpace(Vy, [0])
    Uz = TrialFESpace(Vz, [0, 0])
    P  = TrialFESpace(Q)
    B  = TrialFESpace(D, [0])
    X  = MultiFieldFESpace([Ux, Uy, Uz, P])
    nx = Ux.space.nfree
    ny = Uy.space.nfree
    nz = Uz.space.nfree
    nu = nx + ny + nz
    np = P.space.space.nfree
    nb = B.space.nfree
    N = nu + np - 1
    @printf("\nN = %d (%d + %d) ∼ 10^%d DOF\n", N, nu, np-1, floor(log10(N)))

    # # filename for LHS matrix
    # LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%sy_%e_%e_%e_%e_%e_%e.h5", dim, hres, ε², γ, f₀, β, k)

    a((ux, uy, uz, p), (vx, vy, vz, q)) =
        ∫( -ux*vx*k^2*γ*  ε²*ν + ∂y(ux)*∂y(vx)*γ*  ε²*ν + ∂z(ux)*∂z(vx)*  ε²*ν - uy*vx*f + p*vx*im*k +
           -uy*vy*k^2*γ*  ε²*ν + ∂y(uy)*∂y(vy)*γ*  ε²*ν + ∂z(uy)*∂z(vy)*  ε²*ν + ux*vy*f + ∂y(p)*vy +
           -uz*vz*k^2*γ^2*ε²*ν + ∂y(uz)*∂y(vz)*γ^2*ε²*ν + ∂z(uz)*∂z(vz)*γ*ε²*ν +           ∂z(p)*vz +
            ux*q*im*k + ∂y(uy)*q + ∂z(uz)*q )dΩ
    @time "assemble LHS_inversion" LHS_inversion = assemble_matrix(a, X, Y)
    # @time "RCM perm" perm_inversion, inv_perm_inversion = nuPGCM.RCM_perm(arch, X, Y, dΩ)
    # LHS_inversion = LHS_inversion[perm_inversion, perm_inversion]
    # write_sparse_matrix(LHS_inversion, perm_inversion, inv_perm_inversion; fname=LHS_inversion_fname)

    # lu factor 
    @time "lu(A)" LHS_inversion = lu(LHS_inversion)

    # inversion RHS
    a_rhs(b, vz) = ∫( b*vz )dΩ
    @time "RHS_inversion" RHS_inversion = assemble_matrix(a_rhs, B, Vz)

    # inversion function w = L(b)
    function L(b)
        RHS = [zeros(nx); zeros(ny); RHS_inversion*b; zeros(np-1)]
        sol = LHS_inversion \ RHS
        return sol[nx+ny+1:nx+ny+nz]
    end

    # DOF
    nb = B.space.nfree
    println("DOF: ", nb)

    return L, X, Y, B, D, Ω, dΩ, nb
end

# assemble
function assemble_system()
    # assemble K
    a_K(b, d) = ∫( -b*d*γ*k^2*κ + ∂y(b)*∂y(d)*γ*κ + ∂z(b)*∂z(d)*κ )dΩ
    @time "assemble K" K = assemble_matrix(a_K, B, D)

    # assemble M
    a_M(b, d) = ∫( b*d )dΩ
    @time "assemble M" M = assemble_matrix(a_M, B, D)

    # assemble M_w and M_wb
    _, _, Uz, _ = unpack_spaces(X)
    _, _, Vz, _ = unpack_spaces(Y)
    @time "assemble M_w" M_w = assemble_matrix(a_M, Uz, Vz)
    @time "assemble M_wb" M_wb = assemble_matrix(a_M, Uz, D)

    # # Cuthill-McKee DOF reordering
    # perm = nuPGCM.RCM_perm(arch, M)
    # inv_perm = invperm(perm)
    # K = K[perm, perm]
    # M = M[perm, perm]
    # perm_w = nuPGCM.RCM_perm(arch, M_w)
    # M_wb = M_wb[perm, perm_w]

    return K, M, M_wb#, inv_perm
end

# load model
L, X, Y, B, D, Ω, dΩ, nb = load_model()

# # debug
# b = interpolate_everywhere(x->0.1*exp(-(x[3] + H(x))/0.1), B)
# _, _, Uz, _ = unpack_spaces(X)
# w = FEFunction(Uz, L(b.free_values))
# plot_slice(real(w), real(b), 1; x=0.0, cb_label=L"w", fname=@sprintf("%s/images/w_xslice.png", out_folder))

# assemble system
# K, M, M_wb, inv_perm = assemble_system()
K, M, M_wb = assemble_system()
@time "lu(M)" M = lu(M)

# solve B^-1 A X = ω X where A = -i*ε²/μϱ*K - i*M_wb*L and B = M
# which = EigSorter(abs; rev = false) # smallest magnitude
which = EigSorter(x -> abs(imag(x)); rev = false) # smallest mag imaginary part
@time vals, vecs, info = eigsolve(x -> M\(-im*ε²/μϱ*K*x - im*M_wb*L(x)), nb, 1, which, ComplexF64, verbosity=2, maxiter=300)
fname = @sprintf("%s/data/eigs_k%0.2f_beta%0.1f.jld2", out_folder, k, β)
jldsave(fname; vals, vecs)
println(fname)

ω = vals[1]
if imag(ω) >= 0
    @printf("ω = %e + %e i\n", real(ω), imag(ω))
else
    @printf("ω = %e - %e i\n", real(ω), -imag(ω))
end
# b = vecs[1][inv_perm]
b = vecs[1]
# b /= 20*maximum(abs.(b)) # scale for plotting
b = FEFunction(B, b)
fig, ax = plt.subplots(1, 3, figsize=(39*pc, 39*pc/1.62/2))
nuPGCM.plot_slice_wave(b, b, 1, k, ω; x=+0.0, cb_label=L"Re $b$", fig=fig, ax=ax[1])
nuPGCM.plot_slice_wave(b, b, 1, k, ω; y=+0.0, cb_label=L"Re $b$", fig=fig, ax=ax[2])
nuPGCM.plot_slice_wave(b, b, 1, k, ω; z=-0.5, cb_label=L"Re $b$", fig=fig, ax=ax[3])
ax[1].set_title(L"(a) $x = 0$")
ax[2].set_title(L"(b) $y = 0$")
ax[3].set_title(L"(c) $z = -0.5$")
ax[2].annotate(latexstring(@sprintf("\$\\omega = %0.2f - %0.2f i\$", real(ω), -imag(ω))), xy=(0.2, 0.75), xycoords="axes fraction")
subplots_adjust(wspace=0.4)
savefig(@sprintf("%s/images/b_k%0.2f_beta%0.2f.png", out_folder, k, β))
println(@sprintf("%s/images/b_k%0.2f_beta%0.2f.png", out_folder, k, β))
plt.close()

function varying_beta()
    fig, ax = plt.subplots(1, 3, figsize=(39*pc, 39*pc/1.62/2))
    betas = [0.1, 0.5, 1.0]
    for i in 1:3
        file = jldopen(@sprintf("%s/data/eigs_k%0.2f_beta%0.1f.jld2", out_folder, k, betas[i]), "r")
        ω = file["vals"][1]
        b = file["vecs"][1]
        close(file)
        b = FEFunction(B, b)
        nuPGCM.plot_slice_wave(b, b, 1, k, ω; x=+0.0, cb_label=L"Re $b$", fig=fig, ax=ax[i])
        ax[i].annotate(latexstring(@sprintf("\$\\omega = %0.3f - %0.3f i\$", real(ω), -imag(ω))), xy=(0.2, 0.75), xycoords="axes fraction")
    end
    ax[1].set_title(L"(a) $\beta = 0.1$")
    ax[2].set_title(L"(b) $\beta = 0.5$")
    ax[3].set_title(L"(c) $\beta = 1$")
    subplots_adjust(wspace=0.4, hspace=0.0)
    savefig(@sprintf("%s/images/varying_beta.png", out_folder))
    println(@sprintf("%s/images/varying_beta.png", out_folder))
    plt.close()
end

# varying_beta()

function disp()
    fig, ax = plt.subplots(1)
    ax.set_xlabel(L"k")
    ax.set_xlim(-2*pi, 0.02)
    ax.set_ylim(0, 0.04)
    ax.set_xticks([-2*pi, -pi, 0])
    ax.set_xticklabels([L"$-2\pi$", L"$-\pi$", L"0"])
    ax.spines["left"].set_visible(false)
    ax.axvline(0, color="black", linewidth=0.5)

    N = 100
    ks = [-2*pi*i/100 for i in 0:N]
    for beta = [0.5, 1.0]
        ωs = zeros(ComplexF64, N+1) 
        for i in eachindex(ks)
            try
                fname = @sprintf("%s/data/eigs_k%0.2f_beta%0.1f.jld2", out_folder, ks[i], beta)
                jldopen(fname, "r") do file
                    ωs[i] = file["vals"][1]
                end
            catch
                ωs[i] = NaN
            end
        end

        c = beta == 0.5 ? "C0" : "C1"
        ax.plot(ks, real.(ωs),  c=c, "-",  label=L"Re $\omega$, $\beta = $"*@sprintf("%0.1f", beta))
        ax.plot(ks, -imag.(ωs), c=c, "--", label=L"$-$Im $\omega$, $\beta = $"*@sprintf("%0.1f", beta))
    end
    ax.legend(ncol=2)
    savefig(@sprintf("%s/images/omega.png", out_folder))
    println(@sprintf("%s/images/omega.png", out_folder))
    plt.close()
end

# disp()

println("Done.")