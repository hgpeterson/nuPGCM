struct InversionToolkit{B, V, S<:IterativeSolverToolkit}
    B::B       # RHS matrix
    b::V       # RHS vector
    solver::S  # iterative solver toolkit
end

function Base.summary(inversion::InversionToolkit)
    t = typeof(inversion)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, inversion::InversionToolkit)
    println(io, summary(inversion), ":")
    println(io, "├── B: ", summary(inversion.B))
    println(io, "├── b: ", summary(inversion.b))
      print(io, "└── solver: ", summary(inversion.solver))
end

"""
    inversion_toolkit = InversionToolkit(arch::AbstractArchitecture, 
                                         fe_data::FEData, 
                                         params::Parameters, 
                                         forcings::Forcings; 
                                         kwargs...)

Set up the inversion toolkit, which contains the matrices and solvers for the inversion problem.                                    
"""
function InversionToolkit(arch::AbstractArchitecture,
                          fe_data::FEData, 
                          params::Parameters, 
                          forcings::Forcings; 
                          kwargs...)
    # build
    @info "Building inversion system..."
    A, B, b = build_inversion_system(fe_data, params, forcings)

    # re-order dofs
    A = A[fe_data.dofs.p_inversion, fe_data.dofs.p_inversion]
    B = B[fe_data.dofs.p_inversion, :]
    b = b[fe_data.dofs.p_inversion]

    # preconditioner
    if typeof(arch) == GPU || forcings.eddy_param.is_on
        # get resolution
        p, t = get_p_t(fe_data.mesh.model)
        edges, _, _ = all_edges(t)
        hs = [norm(p[edges[i, 1], :] - p[edges[i, 2], :]) for i ∈ axes(edges, 1)]
        sort!(hs)
        h = hs[length(hs) ÷ 2] # median edge length
        @debug @sprintf("Median edge length h = %.2e", h)
        dim = size(t, 2) - 1
        @debug "Mesh dimension: $dim"

        # use diagonal preconditioner scaled by resolution
        P = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A, 1))))
    else
        # on CPU and fixed ν → can just LU factor
        @warn "LU-factoring inversion matrix with $(length(fe_data.dofs.p_inversion)) DOFs..."
        @time "lu(A_inversion)" P = lu(A)
    end

    # move to arch
    A = on_architecture(arch, A)
    B = on_architecture(arch, B)
    b = on_architecture(arch, b)

    if typeof(arch) == GPU
        CUDA.memory_status()
    end

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, A, P, B, b; kwargs...)

    return inversion_toolkit
end

function InversionToolkit(arch::AbstractArchitecture, 
                          A, P, B, b; 
                          atol=1e-6, rtol=1e-6, itmax=0, memory=20, history=true, verbose=false, restart=true)
    # rhs vector for solver
    N = size(A, 1)
    T = eltype(A)
    y = on_architecture(arch, zeros(T, N))

    # use GMRES solver (which requires the `memory` and `restart` parameters)
    VT = vector_type(arch, T)
    solver = Krylov.GmresSolver(N, N, memory, VT)
    solver.x .= zero(T)

    # set up keyword arguments for iterative solver toolkit
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int, :restart=>restart)

    solver_inversion = IterativeSolverToolkit(A, P, y, solver, kwargs, "Inversion") 

    return InversionToolkit(B, b, solver_inversion)
end

"""
    invert!(inversion::InversionToolkit, b)

Perform the inversion given buoyancy `b`.
"""
function invert!(inversion::InversionToolkit, b)
    # calculate rhs vector
    arch = architecture(inversion.B)
    inversion.solver.y .= inversion.B*on_architecture(arch, b.free_values) .+ inversion.b

    # solve
    iterative_solve!(inversion.solver)

    return inversion
end

####
#### Matrix-building functions
####

"""
    A, B, b = build_inversion_system(fe_data::FEData, params::Parameters, forcings::Forcings) 

Build the matrices and vectors for the inversion problem of the PG equations.
"""
function build_inversion_system(fe_data::FEData, params::Parameters, forcings::Forcings) 
    A_inversion = build_A_inversion(fe_data, params, forcings.ν)
    B_inversion = build_B_inversion(fe_data, params)
    b_inversion = build_b_inversion(fe_data, params, forcings)
    return A_inversion, B_inversion, b_inversion
end

"""
    A = build_A_inversion(fe_data::FEData, params::Parameters, ν)

Assemble the LHS matrix `A` for the inversion problem. 
"""
function build_A_inversion(fe_data::FEData, params::Parameters, ν; friction_only=false, frictionless_only=false) 
    # unpack
    X_trial = fe_data.spaces.X_trial
    X_test = fe_data.spaces.X_test
    dΩ = fe_data.mesh.dΩ
    α²ε² = params.α^2*params.ε^2
    f = params.f

    # bilinear form
    a((u, p), (v, q)) = bilinear_form((u, p), (v, q), α²ε², f, ν, dΩ; friction_only, frictionless_only)

    # assemble 
    @time "build inversion system" A = assemble_matrix(a, X_trial, X_test)

    return A
end
function bilinear_form((u, p), (v, q), α²ε², f, ν, dΩ; friction_only, frictionless_only)
    σ = Gridap.symmetric_gradient
    # for general ν, need full stress tensor
    if friction_only
        return ∫( 2*α²ε²*(ν*σ(u)⊙σ(v)) )*dΩ
    elseif frictionless_only
        return ∫( -(∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    else
        return ∫( 2*α²ε²*(ν*σ(u)⊙σ(v)) - (∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    end
end
function bilinear_form((u, p), (v, q), α²ε², f, ν::Real, dΩ; friction_only, frictionless_only)
    # since ν is constant, we can just use the Laplacian here
    if friction_only
        return ∫( α²ε²*(ν*∇(u)⊙∇(v)) )*dΩ
    elseif frictionless_only
        return ∫( -(∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    else
        return ∫( α²ε²*(ν*∇(u)⊙∇(v)) - (∇⋅v)*p + q*(∇⋅u) + f*((z⃗×u)⋅v) )*dΩ
    end
end

"""
    B = build_B_inversion(fe_data::FEData, params::Parameters)

Assemble the RHS matrix for the inversion problem.
"""
function build_B_inversion(fe_data::FEData, params::Parameters)
    # unpack
    U_test = fe_data.spaces.X_test[1]
    B_trial = fe_data.spaces.B_trial
    dΩ = fe_data.mesh.dΩ
    α = params.α

    # bilinear form
    a(b, v) = ∫( 1/α*(b*(z⃗⋅v)) )dΩ

    # assemble
    B = assemble_matrix(a, B_trial, U_test) 

    # convert to N × nb matrix
    nu, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + np
    I, J, V = findnz(B)
    B = sparse(I, J, V, N, nb)

    return B
end

"""
    b = build_b_inversion(mesh::FEData, params::Parameters, forcings::Forcings)

Assemble the RHS vector for the inversion problem.
"""
function build_b_inversion(fe_data::FEData, params::Parameters, forcings::Forcings)
    # unpack
    U_test  = fe_data.spaces.X_test[1]
    b_diri = fe_data.spaces.b_diri
    dΓ = fe_data.mesh.dΓ
    dΩ = fe_data.mesh.dΩ
    α = params.α
    τˣ = forcings.τˣ
    τʸ = forcings.τʸ

    # allocate vector of length N
    nu, np, nb = get_n_dofs(fe_data.dofs)
    N = nu + np
    b = zeros(N)

    # linear form
    l(v) = ∫( α*(τˣ*(x⃗⋅v) + τʸ*(y⃗⋅v)) )dΓ + # b.c. is α²ε²ν∂z(u) = ατ
           ∫( 1/α*(b_diri*(z⃗⋅v)) )dΩ        # correction due to Dirichlet boundary condition

    # assemble
    b[1:nu] .= assemble_vector(l, U_test)

    return b
end