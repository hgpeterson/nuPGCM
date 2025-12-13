mutable struct InversionToolkit{B, V, S<:IterativeSolverToolkit}
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

    # # preconditioner
    # if typeof(arch) == GPU || forcings.eddy_param.is_on
    #     # get resolution
    #     p, t = get_p_t(fe_data.mesh.model)
    #     edges, _, _ = all_edges(t)
    #     hs = [norm(p[edges[i, 1], :] - p[edges[i, 2], :]) for i ∈ axes(edges, 1)]
    #     sort!(hs)
    #     h = hs[length(hs) ÷ 2] # median edge length
    #     @debug @sprintf("Median edge length h = %.2e", h)
    #     dim = size(t, 2) - 1
    #     @debug "Mesh dimension: $dim"

    #     # use diagonal preconditioner scaled by resolution
    #     P = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A, 1))))
    # else
    #     # # on CPU and fixed ν → can just LU factor
    #     # @warn "LU-factoring inversion matrix with $(length(fe_data.dofs.p_inversion)) DOFs..."
    #     # @time "lu(A_inversion)" P = lu(A)
    #     P = BlockDiagonalPreconditioner(arch, params, fe_data, A)
    # end
    P = BlockDiagonalPreconditioner(arch, params, fe_data, A)

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