using Test
using nuPGCM
using Gridap
using CUDA
using SparseArrays
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

set_out_dir!("./test")

function coarse_inversion(dim, arch)
    # params/funcs
    ε² = 1e-2
    γ = 1/4
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = 1 - x[1]^2 - x[2]^2
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

    # coarse mesh
    h = 0.1
    mesh = Mesh(@sprintf("meshes/bowl%s_%0.2f.msh", dim, h))

    # dof
    nu, nv, nw, np, nb = get_n_dof(mesh)
    if typeof(dim) == TwoD
        @test nu + nv + nw + np == 2023
    else
        @test nu + nv + nw + np == 27934
    end

    # dof perms
    p_u, p_v, p_w, p_p, p_b = nuPGCM.compute_dof_perms(mesh)
    inv_p_b = invperm(p_b)
    p_inversion = [p_u; p_v .+ nu; p_w .+ nu .+ nv; p_p .+ nu .+ nv .+ nw]
    inv_p_inversion = invperm(p_inversion)

    # assemble LHS inversion and test against saved matrix
    A_inversion_fname = @sprintf("test/data/A_inversion_%s_%e_%e_%e_%e_%e.h5", dim, h, ε², γ, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion = nuPGCM.build_A_inversion(mesh, γ, ε², ν, f; fname=A_inversion_fname)
    else
        A_inversion = nuPGCM.build_A_inversion(mesh, γ, ε², ν, f; fname="A_inversion_temp.jld2")
        jldopen(A_inversion_fname, "r") do file
            @test A_inversion ≈ file["A_inversion"]
        end
    end

    # re-order dof
    A_inversion = A_inversion[p_inversion, p_inversion]

    # build RHS matrix for inversion
    B_inversion = nuPGCM.build_B_inversion(mesh)

    # re-order dof
    B_inversion = B_inversion[p_inversion, :]

    # preconditioner
    if typeof(arch) == CPU
        P_inversion = lu(A_inversion)
    else
        P_inversion = Diagonal(on_architecture(arch, 1/h^dim.n*ones(N)))
    end

    # move to arch
    A_inversion = on_architecture(arch, A_inversion)
    B_inversion = on_architecture(arch, B_inversion)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion)

    function update_u_p!(u, v, w, p, solver)
        sol = on_architecture(CPU(), solver.x[inv_p_inversion])
        u.free_values .= sol[1:nu]
        v.free_values .= sol[nu+1:nu+nv]
        w.free_values .= sol[nu+nv+1:nu+nv+nw]
        p = FEFunction(P, sol[nu+nv+nw+1:end])
        return u, v, w, p
    end

    # background state ∂z(b) = N^2
    N² = 1.

    # simple test buoyancy field: b = δ exp(-(z + H)/δ)
    U, V, W, P = unpack_spaces(mesh.X_trial)
    B = mesh.B_trial
    b = interpolate_everywhere(x -> 0.1*exp(-(x[3] + H(x))/0.1), B)
    u = interpolate_everywhere(0, U)
    v = interpolate_everywhere(0, V)
    w = interpolate_everywhere(0, W)
    p = interpolate_everywhere(0, P) 

    # invert
    invert!(inversion_toolkit, b)
    u, v, w, p = update_u_p!(u, v, w, p, inversion_toolkit.solver)

    # plot for sanity check
    sim_plots(dim, u, v, w, b, N², H, 0, 0)

    # compare state with data
    datafile = @sprintf("test/data/inversion_%s.h5", dim)
    if !isfile(datafile)
        @warn "Data file not found, saving state..."
        save_state(u, v, w, p, b, 0; fname=datafile)
    else
        u_data, v_data, w_data, p_data, b_data, t_data = load_state(@sprintf("test/data/inversion_%s.h5", dim))
        @test isapprox(u.free_values, u_data, rtol=1e-2)
        @test isapprox(v.free_values, v_data, rtol=1e-2)
        @test isapprox(w.free_values, w_data, rtol=1e-2)
        @test isapprox(p.free_values, p_data, rtol=1e-2)
        @test isapprox(b.free_values, b_data, rtol=1e-2)
    end

    # remove temporary files
    rm("A_inversion_temp.jld2", force=true)
end

@testset "Inversion Tests" begin
    @testset "2D CPU" begin
        coarse_inversion(TwoD(), CPU())
    end
    # @testset "2D GPU" begin
    #     coarse_inversion(TwoD(), GPU())
    # end
    # @testset "3D CPU" begin
    #     coarse_inversion(ThreeD(), CPU())
    # end
    # @testset "3D GPU" begin
    #     coarse_inversion(ThreeD(), GPU())
    # end
end