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

    # derivative matrices
    Dξ::SparseMatrixCSC{Float64,Int64}
    Dσ::SparseMatrixCSC{Float64,Int64}

    # inversion matrices
    inversionLHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}

    # U = 1 solution
    sol_U::Array{Float64,2}
end

# construct ModelSetup object using analytical functions for H, Hx, ν, κ
function ModelSetup(f::Float64, N::Float64, ξVariation::Bool, L::Float64, nξ::Int64, nσ::Int64, H_func::Function, Hx_func::Function, ν_func::Function, κ_func::Function, Δt::Real)
    # create grids: even spacing in ξ and chebyshev in σ
    ξ = collect(0:L/nξ:(L - L/nξ))
    σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    # σ = collect(-1:1/(nσ - 1):0)

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

    # get derivative matrices
    Dξ, Dσ = getDerivativeMatrices(ξ, σ, L)
    
    # inversion matrices
    inversionLHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, nξ) 
    for i=1:nξ 
        inversionLHSs[i] = lu(getInversionLHS(ν[i, :], f, H[i], σ)) 
    end  
    
    # U = 1 solution  
    inversionRHS = getInversionRHS(zeros(nξ, nσ), 1) 
    sol_U = computeSol(inversionLHSs, inversionRHS) 

    return ModelSetup(f, N, ξVariation, L, nξ, nσ, ξ, σ, x, z, H, Hx, ν, κ, Δt, Dξ, Dσ, inversionLHSs, sol_U)
end

"""
    Dξ, Dσ = getDerivativeMatrices(ξ, σ, L)

Compute the derivative matrices.
"""
function getDerivativeMatrices(ξ::Array{Float64,1}, σ::Array{Float64,1}, L::Float64)
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
                # left (periodic)
                fd_ξ = mkfdstencil([ξ[nξ] - L, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (row, umap[nξ, j], fd_ξ[1]))
                push!(Dξ, (row, umap[1, j],  fd_ξ[2]))
                push!(Dξ, (row, umap[2, j],  fd_ξ[3]))
            elseif i == nξ
                # right (periodic)
                fd_ξ = mkfdstencil([ξ[nξ-1], ξ[nξ], ξ[1] + L], ξ[nξ], 1)
                push!(Dξ, (row, umap[nξ-1, j], fd_ξ[1]))
                push!(Dξ, (row, umap[nξ, j],   fd_ξ[2]))
                push!(Dξ, (row, umap[1, j],    fd_ξ[3]))
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
