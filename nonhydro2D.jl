using NonhydroPG
using Gridap
using GridapGmsh
using Gmsh: gmsh
using Printf
using LinearAlgebra
using IterativeSolvers
using HDF5
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function run()
    # model
    # model = GmshDiscreteModel("bowl2D.msh")
    model = GmshDiscreteModel("mesh.msh")
    # model = GmshDiscreteModel("pydistmesh/mesh_uniform_0.01.msh")
    # model = GmshDiscreteModel("pydistmesh/mesh_linear_0.01_2_0.3.msh")
    # model = GmshDiscreteModel("pydistmesh/mesh_exp_0.005_4_0.1.msh")

    # for plotting
    g = MyGrid(model)

    # depth
    # H(x) = 1 - x[1]^2
    H(x) = 1 - (x[1]/L)^2

    # reference FE 
    order = 2
    reffe_ux = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_uy = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_uz = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_p  = ReferenceFE(lagrangian, Float64, order-1; space=:P)
    reffe_b  = ReferenceFE(lagrangian, Float64, order;   space=:P)

    # test FESpaces
    Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot"])
    Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot"])
    Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
    Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
    D  = TestFESpace(model, reffe_b,  conformity=:H1, dirichlet_tags=["sfc"])
    Y  = MultiFieldFESpace([Vx, Vy, Vz, Q])

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

    # triangulation and integration measure
    degree = order^2
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # gradients 
    ∂x(u) = VectorValue(1.0, 0.0)⋅∇(u)
    ∂z(u) = VectorValue(0.0, 1.0)⋅∇(u)

    # forcing
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[2] + H(x))/0.1)

    # params
    ε² = 1e-4
    println("δ = ", √(2ε²))
    pts, conns = get_p_t(model)
    h1 = [norm(pts[conns[i, 1], :] - pts[conns[i, 2], :]) for i ∈ axes(conns, 1)]
    h2 = [norm(pts[conns[i, 2], :] - pts[conns[i, 3], :]) for i ∈ axes(conns, 1)]
    h3 = [norm(pts[conns[i, 3], :] - pts[conns[i, 1], :]) for i ∈ axes(conns, 1)]
    hmin = minimum([h1; h2; h3])
    hmax = maximum([h1; h2; h3])
    println("hmin = ", hmin)
    println("hmax = ", hmax)
    μϱ = 1e0
    γ = 1
    f = 1
    Δt = 1e-2

    # bilinear form for inversion
    ainv((ux, uy, uz, p), (vx, vy, vz, q)) = 
        # ∫(γ*ε²*∂x(ux)*∂x(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - f*uy*vx + ∂x(p)*vx +
        #   γ*ε²*∂x(uy)*∂x(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + f*ux*vy            +
        # γ*γ*ε²*∂x(uz)*∂x(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
        #                          ∂x(ux)*q + ∂z(uz)*q )dΩ
        ∫(   ε²*∂z(ux)*∂z(vx)*ν - f*uy*vx + ∂x(p)*vx +
             ε²*∂z(uy)*∂z(vy)*ν + f*ux*vy            +
           γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                 ∂x(ux)*q + ∂z(uz)*q )dΩ
    Ainv = assemble_matrix(ainv, X, Y)
    Ainv_factored = lu(Ainv)

    # inversion function
    # sol = zeros(size(Ainv, 2))
    # function invert!(sol, b)
    #     linv((vx, vy, vz, q)) = ∫( b*vz )dΩ
    #     rhsinv = assemble_vector(linv, Y)
    #     # sol = Ainv_factored \ rhsinv
    #     @time bicgstabl!(sol, Ainv, rhsinv)
    # end
    # function update_state!(ux, uy, uz, p, sol)
    #     ux.free_values .= sol[1:nx]
    #     uy.free_values .= sol[nx+1:nx+ny]
    #     uz.free_values .= sol[nx+ny+1:nx+ny+nz]
    #     p = FEFunction(P, sol[nx+ny+nz+1:end])
    #     return ux, uy, uz, p
    # end
    function invert!(ux, uy, uz, p, b)
        linv((vx, vy, vz, q)) = ∫( b*vz )dΩ
        rhsinv = assemble_vector(linv, Y)
        sol = Ainv_factored \ rhsinv
        ux.free_values .= sol[1:nx]
        uy.free_values .= sol[nx+1:nx+ny]
        uz.free_values .= sol[nx+ny+1:nx+ny+nz]
        p = FEFunction(P, sol[nx+ny+nz+1:end])
        return ux, uy, uz, p
    end

    # initial condition
    b0(x) = x[2]
    # b0(x) = x[2] + 0.1*exp(-(x[2] + H(x))/0.1)
    b  = interpolate_everywhere(b0, B)
    ux = interpolate_everywhere(0, Ux)
    uy = interpolate_everywhere(0, Uy)
    uz = interpolate_everywhere(0, Uz)
    p  = interpolate_everywhere(0, P)
    ux, uy, uz, p = invert!(ux, uy, uz, p, b)
    i_img = 0
    plots(g, ux, uy, uz, p, b, i_img)
    # writevtk(Ω, @sprintf("out/nonhydro2D%03d", i_img), cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])
    profile_plot(ux, uy, uz, b, L/2, 0.0, H, @sprintf("images/profiles%03d.png", i_img))
    i_img += 1
    # error("stop")

    # b^n+1 - Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n+1)) = b^n - Δt*u^n⋅∇b^n + Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n))
    # evolution LHS
    alhs(b, d) = ∫( b*d + Δt/2*ε²/μϱ*∂z(b)*∂z(d)*κ )dΩ
    LHS = assemble_matrix(alhs, B, D)
    LHS_factored = lu(LHS)
    for i ∈ 1:1000
        # evolution RHS
        lrhs(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uz*∂z(b)*d - Δt/2*ε²/μϱ*∂z(b)*∂z(d)*κ )dΩ
        rhs = assemble_vector(lrhs, D)
        b.free_values .= LHS_factored \ rhs
        # @time cg!(b.free_values, LHS, rhs)
        ux, uy, uz, p = invert!(ux, uy, uz, p, b)
        # invert!(sol, b)
        # ux, uy, uz, p = update_state!(ux, uy, uz, p, sol)
        if mod(i, 10) == 0
            @printf("% 5d  %1.2e\n", i, min(hmin/maximum(abs.(ux.free_values)), hmin/maximum(abs.(uz.free_values))))
        end
        if mod(i, 100) == 0
            plots(g, ux, uy, uz, p, b, i_img)
            # writevtk(Ω, @sprintf("out/nonhydro2D%03d", i_img), cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])
            # profile_plot(ux, uy, uz, b, 0.5, i*Δt, H, @sprintf("images/profiles%03d.png", i_img))
            profile_plot(ux, uy, uz, b, L/2, i*Δt, H, @sprintf("images/profiles%03d.png", i_img))
            i_img += 1
        end
    end

    return ux, uy, uz, p, b
end

function plots(g, ux, uy, uz, p, b, i)
    quick_plot(ux, g, b=b, label=L"u", fname=@sprintf("images/u%03d.png", i))
    quick_plot(uy, g, b=b, label=L"v", fname=@sprintf("images/v%03d.png", i))
    quick_plot(uz, g, b=b, label=L"w", fname=@sprintf("images/w%03d.png", i))
    quick_plot(p,  g, b=b, label=L"p", fname=@sprintf("images/p%03d.png", i))
end

function profile_plot(ux, uy, uz, b, x, t, H, fname)
    z = H([x])*(chebyshev_nodes(2^6) .- 1)/2

    uxs = [nan_eval(ux, Point(x, zᵢ)) for zᵢ ∈ z]
    uys = [nan_eval(uy, Point(x, zᵢ)) for zᵢ ∈ z]
    uzs = [nan_eval(uz, Point(x, zᵢ)) for zᵢ ∈ z]
    bz = VectorValue(0.0, 1.0)⋅∇(b)
    bzs = [nan_eval(bz, Point(x, zᵢ)) for zᵢ ∈ z]
    uxs[1] = 0
    uys[1] = 0
    uzs[1] = 0
    bzs[1] = 0

    # gamma = 0 solution
    # file = h5open("gamma0_Ek1e-3.h5", "r")
    file = h5open("gamma0_Ek1e-4.h5", "r")
    u0 = read(file, "u")
    v0 = read(file, "v")
    w0 = read(file, "w")
    bz0 = read(file, "bz")
    z0 = read(file, "z")
    close(file)

    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2))
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    ax[4].set_xlabel(L"\partial_z b")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    for a ∈ ax 
        a.set_ylim(-H([x]), 0) 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2,2))
    end
    ax[1].plot(u0,  z0, label=L"\gamma = 0")
    ax[2].plot(v0,  z0)
    ax[3].plot(w0,  z0)
    ax[4].plot(bz0, z0)
    ax[1].plot(uxs, z, label=L"\gamma = 1/8")
    ax[2].plot(uys, z)
    ax[3].plot(uzs, z)
    ax[4].plot(bzs, z)
    ax[1].set_title(L"x = "*@sprintf("%1.2f", x)*L", \quad t = "*@sprintf("%1.2f", t))
    # ax[1].set_title(L"x = "*@sprintf("%1.2f", x))
    ax[1].legend()
    savefig(fname)
    println(fname)
    plt.close()
end

L = 1
ux, uy, uz, p, b = run()

# model = GmshDiscreteModel("bowl2D.msh")
# g = MyGrid(model)
# profile_plot(ux, uy, uz, b, 0.5, 10.0, x->1-x[1]^2, "images/profiles010.png")
