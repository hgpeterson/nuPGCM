using NonhydroPG
using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot
using Printf

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function run()
    # model
    model = GmshDiscreteModel("bowl2D.msh")
    writevtk(model, "model")

    # for plotting
    g = MyGrid(model)

    # reference FE 
    order = 2
    refFE = ReferenceFE(lagrangian, Float64, order; space=:P)

    # test FESpace
    Φ = TestFESpace(model, refFE, conformity=:H1, dirichlet_tags=["bot", "sfc"])
    P = TestFESpace(model, refFE, conformity=:H1)

    # trial FESpace with Dirichlet values
    Ψ = TrialFESpace(Φ, [0, 0])
    Q = TrialFESpace(P)

    # triangulation and integration measure
    degree = order^2
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # params
    Δ = 0.1 
    H0 = -0.5
    L0 = 0.2
    Δt = 5e-2

    # gradients
    ∂x(u) = ∇(u)⋅VectorValue(1, 0)
    ∂y(u) = ∇(u)⋅VectorValue(0, 1)

    # bilinear form for inversion ∇²ψ = q
    ainv(ψ, ϕ) = ∫( -∇(ψ)⋅∇(ϕ) )dΩ
    Ainv = assemble_matrix(ainv, Ψ, Φ)
    Ainv_factored = numerical_setup(symbolic_setup(BackslashSolver(), Ainv), Ainv)

    # mass matrix for inversion
    amminv(q, ϕ) = ∫( q*ϕ )dΩ
    MMinv = assemble_matrix(amminv, Q, Φ)

    # inversion function
    function invert!(ψ, q)
        solve!(ψ, Ainv_factored, MMinv*q)
    end

    # mass matrix for timestepping
    amm(q, p) = ∫( q*p )dΩ
    MM = assemble_matrix(amm, Q, P)
    MM_factored = numerical_setup(symbolic_setup(BackslashSolver(), MM), MM)

    # initial condition
    q(x) = exp(-((x[1] - L0)^2 + (x[2] - H0)^2)/Δ^2) + exp(-((x[1] + L0)^2 + (x[2] - H0)^2)/Δ^2)
    q = interpolate_everywhere(q, Q)
    ψ = interpolate_everywhere(0, Ψ)
    invert!(ψ.free_values, q.free_values)
    writevtk(Ω, @sprintf("out/qg%04d", 0), cellfields=["ψ"=>ψ, "q"=>q])
    quick_plot(q, g, label=L"q",    fname=@sprintf("images/q%04d.png", 0))
    quick_plot(ψ, g, label=L"\psi", fname=@sprintf("images/psi%04d.png", 0))

    # loop
    dq = similar(q.free_values)
    dq_prev = similar(q.free_values)
    for i ∈ 1:1000
        if norm(q.free_values) > 1e10
            break
        end

        # invert
        invert!(ψ.free_values, q.free_values)

        # u⋅∇q term 
        ladv(p) = ∫( (-∂y(ψ)*∂x(q) + ∂x(ψ)*∂y(q))*p )dΩ
        adv = assemble_vector(ladv, P)
        solve!(dq, MM_factored, adv)

        if i == 1
            # euler first step
            q.free_values .-= Δt*dq
        else
            # AB2 otherwise
            q.free_values .-= 3/2*Δt*dq - 1/2*Δt*dq_prev
        end
        dq_prev .= dq

        if mod(i, 10) == 0
            writevtk(Ω, @sprintf("out/qg%04d", i), cellfields=["ψ"=>ψ, "q"=>q])
            quick_plot(q, g, label=L"q",    fname=@sprintf("images/q%04d.png", i))
            quick_plot(ψ, g, label=L"\psi", fname=@sprintf("images/psi%04d.png", i))
            @printf("% 4d  %1.5e\n", i, norm(q.free_values))
        end
    end

    return ψ, q
end

ψ, q, = run()