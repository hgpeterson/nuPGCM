################################################################################
# Model setup
################################################################################

# using SparseArrays, SuiteSparse, LinearAlgebra
# include("../../myJuliaLib.jl")

################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState
    # buoyancy (m s-2)
	b::Array{Float64,2}

    # streamfunction (m2 s-1)
    χ::Array{Float64,2}

    # velocities (m s-1)
	uξ::Array{Float64,2}
	uη::Array{Float64,2}
	uσ::Array{Float64,2}

    # iteration
    i::Array{Int64,1}
end

struct ModelSetup
	# Coriolis parameter (s-1)
	f::Float64

    # buoyancy frequency (s-1)
	N::Float64

	# turn on/off variations in ξ
	ξVariation::Bool

    # width of domain (m)
	L::Float64

	# number of grid points
	nξ::Int64
	nσ::Int64

    # coordinates
    coords::String

    # periodic in x direction?
    periodic::Bool

	# grid coordinates
	ξ::Array{Float64,1}
	σ::Array{Float64,1}
    x::Array{Float64,2}
    z::Array{Float64,2}

    # depth (m)
    H::Array{Float64, 1}

    # derivative of depth w.r.t. x
    Hx::Array{Float64,1}

    # turbulent viscosity (m2 s-1)
	ν::Array{Float64,2}

    # turbulent diffusivity (m2 s-1)
	κ::Array{Float64,2}

    # timestep (s)
	Δt::Float64

    # derivative matrices
    Dξ::SparseMatrixCSC{Float64,Int64}
    Dσ::SparseMatrixCSC{Float64,Int64}

    # diffusion matrix
    D::SparseMatrixCSC{Float64,Int64}

    # inversion LHSs
    inversionLHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}

    # evolution LHS
    evolutionLHS::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}

    # U = 1 solution
    sol_U::Array{Float64,2}
end

################################################################################
# Initialization
################################################################################

"""
    m = ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, H_func, Hx_func, ν_func, κ_func, Δt)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, and κ.
"""
function ModelSetup(f::Float64, N::Float64, ξVariation::Bool, L::Float64, nξ::Int64, nσ::Int64, coords::String, 
                    periodic::Bool, ξ::Array{Float64,1}, σ::Array{Float64,1}, H_func::Function, Hx_func::Function, 
                    ν_func::Function, κ_func::Function, Δt::Real)
    # evaluate functions 
    H = @. H_func(ξ)
    Hx = @. Hx_func(ξ)
    ν = zeros(nξ, nσ)
    κ = zeros(nξ, nσ)
    for i=1:nξ
        ν[i, :] = @. ν_func(ξ[i], σ)
        κ[i, :] = @. κ_func(ξ[i], σ)
    end

    # 2D coordinates in (x, z)
    x = repeat(ξ, 1, nσ)
    z = repeat(σ', nξ, 1).*repeat(H, 1, nσ)

    # pass to setup for arrays
    return ModelSetup(f, N, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, Δt)
end

"""
    m = ModelSetup(f, N, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, Δt)

Construct a ModelSetup struct using arrays of H, Hx, ν, and κ.
"""
function ModelSetup(f::Float64, N::Float64, ξVariation::Bool, L::Float64, nξ::Int64, nσ::Int64, coords::String, 
                    periodic::Bool, ξ::Array{Float64,1}, σ::Array{Float64,1}, x::Array{Float64,2}, z::Array{Float64,2}, 
                    H::Array{Float64,1}, Hx::Array{Float64,1}, ν::Array{Float64,2}, κ::Array{Float64,2}, Δt::Real)
    # get derivative matrices
    Dξ, Dσ = getDerivativeMatrices(ξ, σ, L, periodic)

    # get diffusion matrix
    D = getDiffusionMatrix(ξ, σ, κ, H)

    # inversion LHSs
    inversionLHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, nξ) 
    for i=1:nξ 
        inversionLHSs[i] = getInversionLHS(ν[i, :], f, H[i], σ)
    end  

    # evolution LHS
    evolutionLHS = getEvolutionLHS(nξ, nσ, D, Δt)
    
    # U = 1 inversion solution  
    inversionRHS = getInversionRHS(zeros(nξ, nσ), 1) 
    sol_U = computeSol(inversionLHSs, inversionRHS) 

    return ModelSetup(f, N, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, Δt, Dξ, Dσ, D, inversionLHSs, evolutionLHS, sol_U)
end

"""
    Dξ, Dσ = getDerivativeMatrices(ξ, σ, L, periodic)

Compute the derivative matrices.
"""
function getDerivativeMatrices(ξ::Array{Float64,1}, σ::Array{Float64,1}, L::Float64, periodic::Bool)
    nξ = size(ξ, 1)
    nσ = size(σ, 1)
    nPts = nξ*nσ

    umap = reshape(1:nPts, nξ, nσ)    
    Dξ = Tuple{Int64,Int64,Float64}[]
    Dσ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for i=1:nξ
        for j=1:nσ
            row = umap[i, j] 

            if j == 1 
                # bottom 
                fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
                push!(Dσ, (row, umap[i, 1], fd_σ[1]))
                push!(Dσ, (row, umap[i, 2], fd_σ[2]))
                push!(Dσ, (row, umap[i, 3], fd_σ[3]))
            elseif j == nσ
                # top 
                fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
                push!(Dσ, (row, umap[i, nσ-2], fd_σ[1]))
                push!(Dσ, (row, umap[i, nσ-1], fd_σ[2]))
                push!(Dσ, (row, umap[i, nσ],   fd_σ[3]))
            else
                # interior
                fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
                push!(Dσ, (row, umap[i, j-1], fd_σ[1]))
                push!(Dσ, (row, umap[i, j],   fd_σ[2]))
                push!(Dσ, (row, umap[i, j+1], fd_σ[3]))
            end

            if i == 1
                # left 
                if periodic
                    fd_ξ = mkfdstencil([ξ[nξ] - L, ξ[1], ξ[2]], ξ[1], 1) 
                    push!(Dξ, (row, umap[nξ, j], fd_ξ[1]))
                    push!(Dξ, (row, umap[1, j],  fd_ξ[2]))
                    push!(Dξ, (row, umap[2, j],  fd_ξ[3]))
                else
                    fd_ξ = mkfdstencil(ξ[1:3], ξ[1], 1) 
                    push!(Dξ, (row, umap[1, j], fd_ξ[1]))
                    push!(Dξ, (row, umap[2, j],  fd_ξ[2]))
                    push!(Dξ, (row, umap[3, j],  fd_ξ[3]))
                end
            elseif i == nξ
                # right
                if periodic
                    fd_ξ = mkfdstencil([ξ[nξ-1], ξ[nξ], ξ[1] + L], ξ[nξ], 1)
                    push!(Dξ, (row, umap[nξ-1, j], fd_ξ[1]))
                    push!(Dξ, (row, umap[nξ, j],   fd_ξ[2]))
                    push!(Dξ, (row, umap[1, j],    fd_ξ[3]))
                else
                    fd_ξ = mkfdstencil(ξ[nξ-2:nξ], ξ[nξ], 1)
                    push!(Dξ, (row, umap[nξ-2, j], fd_ξ[1]))
                    push!(Dξ, (row, umap[nξ-1, j], fd_ξ[2]))
                    push!(Dξ, (row, umap[nξ, j],   fd_ξ[3]))
                end
            else
                # interior
                fd_ξ = mkfdstencil(ξ[i-1:i+1], ξ[i], 1)
                push!(Dξ, (row, umap[i-1, j], fd_ξ[1]))
                push!(Dξ, (row, umap[i, j],   fd_ξ[2]))
                push!(Dξ, (row, umap[i+1, j], fd_ξ[3]))
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dξ = sparse((x->x[1]).(Dξ), (x->x[2]).(Dξ), (x->x[3]).(Dξ), nPts, nPts)
    Dσ = sparse((x->x[1]).(Dσ), (x->x[2]).(Dσ), (x->x[3]).(Dσ), nPts, nPts)

    return Dξ, Dσ
end

"""
    D = getDiffusionMatrix(ξ, σ, κ, H)

Compute the matrices needed for evolution equation integration.
"""
function getDiffusionMatrix(ξ::Array{Float64,1}, σ::Array{Float64,1}, κ::Array{Float64,2},  H::Array{Float64,1})
    nξ = size(ξ, 1)
    nσ = size(σ, 1)
    nPts = nξ*nσ

    umap = reshape(1:nPts, nξ, nσ)    
    D = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix (with boundary flux conditions)

    # Main loop, insert stencil in matrices for each node point
    for i=1:nξ
        # interior nodes only for operators
        for j=2:nσ-1
            row = umap[i, j] 

            # dσ stencil
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            κ_σ = sum(fd_σ.*κ[i, j-1:j+1])

            # dσσ stencil
            fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

            # diffusion term: dσ(κ*dσ(b))/H^2 = 1/H^2*(dσ(κ)*dσ(b) + κ*dσσ(b))
            push!(D, (row, umap[i, j-1], (κ_σ*fd_σ[1] + κ[i, j]*fd_σσ[1])/H[i]^2))
            push!(D, (row, umap[i, j],   (κ_σ*fd_σ[2] + κ[i, j]*fd_σσ[2])/H[i]^2))
            push!(D, (row, umap[i, j+1], (κ_σ*fd_σ[3] + κ[i, j]*fd_σσ[3])/H[i]^2))
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, 1], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, 2], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, 3], fd_σ[3]/H[i]))

        # flux at boundaries: top
        row = umap[i, nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, nσ-2], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, nσ-1], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, nσ],   fd_σ[3]/H[i]))
    end

    # Create CSC sparse matrix from matrix elements
    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), nPts, nPts)

    return D
end

"""
    inversionLHS = getInversionLHS(ν, f, H, σ)

Setup left hand side of linear system for problem.
"""
function getInversionLHS(ν::Array{Float64,1}, f::Float64, H::Float64, σ::Array{Float64,1})
    nσ = size(σ, 1)
    iU = nσ + 1
    A = Tuple{Int64,Int64,Float64}[]  

    # for finite difference on the top and bottom boundary
    fd_bot = mkfdstencil(σ[1:3], σ[1], 1)
    fd_top = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    fd_top_σσ = mkfdstencil(σ[nσ-3:nσ], σ[nσ], 2)

    # Main loop, insert stencil in matrix for each node point
    # Lower boundary conditions 
    # b.c. 1: dσ(χ) = 0
    push!(A, (1, 1, fd_bot[1]))
    push!(A, (1, 2, fd_bot[2]))
    push!(A, (1, 3, fd_bot[3]))
    # b.c. 2: χ = 0 
    push!(A, (2, 1, 1.0))

    # Upper boundary conditions
    # b.c. 1: dσσ(χ) = 0 
    push!(A, (nσ, nσ-3, fd_top_σσ[1]))
    push!(A, (nσ, nσ-2, fd_top_σσ[2]))
    push!(A, (nσ, nσ-1, fd_top_σσ[3]))
    push!(A, (nσ, nσ,   fd_top_σσ[4]))
    # b.c. 2: χ - U = 0
    push!(A, (nσ-1, nσ,  1.0))
    push!(A, (nσ-1, iU, -1.0))

    # Interior nodes
    for j=3:nσ-2
        row = j

        # dσ stencil
        fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
        ν_σ = sum(fd_σ.*ν[j-1:j+1])

        # dσσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        ν_σσ = sum(fd_σσ.*ν[j-1:j+1])

        # dσσσ stencil
        fd_σσσ = mkfdstencil(σ[j-2:j+2], σ[j], 3)

        # dσσσσ stencil
        fd_σσσσ = mkfdstencil(σ[j-2:j+2], σ[j], 4)
        
        # eqtn: dσσ(nu*dσσ(χ))/H^4 + f^2*(χ - U)/nu = dξ(b) - dx(H)*σ*dσ(b)/H
        # term 1 (product rule)
        push!(A, (row, j-1, ν_σσ*fd_σσ[1]/H^4))
        push!(A, (row, j,   ν_σσ*fd_σσ[2]/H^4))
        push!(A, (row, j+1, ν_σσ*fd_σσ[3]/H^4))

        push!(A, (row, j-2, 2*ν_σ*fd_σσσ[1]/H^4))
        push!(A, (row, j-1, 2*ν_σ*fd_σσσ[2]/H^4))
        push!(A, (row, j,   2*ν_σ*fd_σσσ[3]/H^4))
        push!(A, (row, j+1, 2*ν_σ*fd_σσσ[4]/H^4))
        push!(A, (row, j+2, 2*ν_σ*fd_σσσ[5]/H^4))

        push!(A, (row, j-2, ν[j]*fd_σσσσ[1]/H^4))
        push!(A, (row, j-1, ν[j]*fd_σσσσ[2]/H^4))
        push!(A, (row, j,   ν[j]*fd_σσσσ[3]/H^4))
        push!(A, (row, j+1, ν[j]*fd_σσσσ[4]/H^4))
        push!(A, (row, j+2, ν[j]*fd_σσσσ[5]/H^4))
        # term 2
        push!(A, (row, j,   f^2/(ν[j])))
        push!(A, (row, iU, -f^2/(ν[j])))
    end

    # U equation: U = ?
    row = iU
    push!(A, (row, row, 1.0))

    # Create CSC sparse matrix from matrix elements
    inversionLHS = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nσ+1, nσ+1)

    return lu(inversionLHS)
end

"""
    evolutionLHS = getEvolutionLHS(nξ, nσ, D, Δt)

Generate the left-hand side matrix for the evolution problem of the form `I - D*Δt`
and flux boundary conditions on the boundaries.
"""
function getEvolutionLHS(nξ::Int64, nσ::Int64, D::SparseMatrixCSC{Float64,Int64}, Δt::Real)
    # implicit euler
    evolutionLHS = I - D*Δt 

    # bottom and top boundaries in 1D
    umap = reshape(1:nξ*nσ, nξ, nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nσ]

    # no flux boundaries
    evolutionLHS[bottomBdy, :] = D[bottomBdy, :]
    evolutionLHS[topBdy, :] = D[topBdy, :]

    return lu(evolutionLHS)
end