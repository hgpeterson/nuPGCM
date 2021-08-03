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

	# grid coordinates
	ξ::Array{Float64,1}
	σ::Array{Float64,1}
    x::Array{Float64,2}
    z::Array{Float64,2}

    # depth (m) as a function of x
    H::Array{Float64, 1}

    # derivative of depth w.r.t. x as a function of x
    Hx::Array{Float64,1}

    # turbulent viscosity (m2 s-1)
	ν::Array{Float64,2}

    # turbulent diffusivity (m2 s-1)
	κ::Array{Float64,2}

    # timestep (s)
	Δt::Float64

    # # derivative matrices
    # ξDerivative::SparseMatrixCSC{Float64, Int64}
    # σDerivative::SparseMatrixCSC{Float64, Int64}

    # inversion matrices
    inversionLHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}

    # U = 1 solution
    sol_U::Array{Float64,2}
end

# construct ModelSetup object using analytical functions for H, Hx, ν, κ
function ModelSetup(f, N, ξVariation, L, nξ, nσ, H_func, Hx_func, ν_func, κ_func, Δt)
    # create grids: even spacing in ξ and chebyshev in σ
    ξ = 0:L/nξ:(L - L/nξ)
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  

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

    # # get derivative matrices
    # ξDerivative, σDerivative = getDerivativeMatrices(ξ, σ, L)
    
    # inversion matrices
    inversionLHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, nξ) 
    for i=1:nξ 
        inversionLHSs[i] = lu(getInversionLHS(ν[i, :], f, H[i], σ)) 
    end  

    # make placeholder struct to call the `computeSol` function
    m = ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, x, z, H, Hx, ν, κ, Δt, inversionLHSs, zeros(nξ, nσ+1))
    
    # U = 1 solution  
    inversionRHS = getInversionRHS(zeros(nξ, nσ), 1) 
    sol_U = computeSol(m, inversionRHS) 

    return ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, x, z, H, Hx, ν, κ, Δt, inversionLHSs, sol_U)
end

# """
#     ξDerivative, σDerivative = getDerivativeMatrices(ξ, σ, L)

# Compute the derivative matrices.
# """
# function getDerivativeMatrices(ξ, σ, L)
#     nξ = size(ξ, 1)
#     nσ = size(σ, 1)
#     nPts = nξ*nσ

#     umap = reshape(1:nPts, nξ, nσ)    
#     ξDerivative = Tuple{Int64,Int64,Float64}[]
#     σDerivative = Tuple{Int64,Int64,Float64}[]

#     # Main loop, insert stencil in matrices for each node point
#     for i=2:nξ-1
#         # interior nodes
#         for j=2:nσ-1
#             row = umap[i, j] 

#             # stencils
#             fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
#             fd_ξ = mkfdstencil(ξ[i-1:i+1], ξ[i], 1)

#             # add to matrices
#             push!(ξDerivative, (row, umap[i-1, j], fd_ξ[1]))
#             push!(ξDerivative, (row, umap[i, j],   fd_ξ[2]))
#             push!(ξDerivative, (row, umap[i+1, j], fd_ξ[3]))

#             push!(σDerivative, (row, umap[i, j-1], fd_σ[1]))
#             push!(σDerivative, (row, umap[i, j],   fd_σ[2]))
#             push!(σDerivative, (row, umap[i, j+1], fd_σ[3]))
#         end
#     end

#     # boundaries
#     for i=1:nξ
#         # bottom 
#         row = umap[i, 1] 
#         fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
#         push!(σDerivative, (row, umap[i, 1], fd_σ[1]))
#         push!(σDerivative, (row, umap[i, 2], fd_σ[2]))
#         push!(σDerivative, (row, umap[i, 3], fd_σ[3]))
#         # top 
#         row = umap[i, nσ] 
#         fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
#         push!(σDerivative, (row, umap[i, nσ-2], fd_σ[1]))
#         push!(σDerivative, (row, umap[i, nσ-1], fd_σ[2]))
#         push!(σDerivative, (row, umap[i, nσ],   fd_σ[3]))
#     end
#     for j=1:nσ
#         # left 
#         row = umap[1, j] 
#         fd_ξ = mkfdstencil([ξ[nξ] - L, ξ[1], ξ[2]], ξ[1], 1) # periodic
#         push!(ξDerivative, (row, umap[nξ, j], fd_ξ[1]))
#         push!(ξDerivative, (row, umap[1, j],  fd_ξ[2]))
#         push!(ξDerivative, (row, umap[2, j],  fd_ξ[3]))
#         # right 
#         row = umap[nξ, j] 
#         fd_ξ = mkfdstencil([ξ[nξ-1], ξ[nξ], ξ[1] + L], ξ[nξ], 1) # periodic
#         push!(ξDerivative, (row, umap[nξ-1, j], fd_ξ[1]))
#         push!(ξDerivative, (row, umap[nξ, j],   fd_ξ[2]))
#         push!(ξDerivative, (row, umap[1, j],    fd_ξ[3]))
#     end

#     # Create CSC sparse matrix from matrix elements
#     ξDerivative = sparse((x->x[1]).(ξDerivative), (x->x[2]).(ξDerivative), (x->x[3]).(ξDerivative), nPts, nPts)
#     σDerivative = sparse((x->x[1]).(σDerivative), (x->x[2]).(σDerivative), (x->x[3]).(σDerivative), nPts, nPts)

#     return ξDerivative, σDerivative
# end

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

"""
    logParams(ofile, text)

Write `text` to `ofile` and print it.
"""
function logParams(ofile::IOStream, text::String)
    write(ofile, string(text, "\n"))
    println(text)
end
