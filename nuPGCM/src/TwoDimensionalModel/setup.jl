################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState2DPG
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

struct ModelSetup2DPG
    # use BL model or full?
    bl::Bool 

	# Coriolis parameter (s-1)
	f::Float64

	# U = 0 or no?
	no_net_transport::Bool

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

    # buoyancy frequency (s-2)
	N2::Array{Float64,2}

    # timestep (s)
	Δt::Float64

    # derivative matrices
    Dξ::SparseMatrixCSC{Float64,Int64}
    Dσ::SparseMatrixCSC{Float64,Int64}

    # diffusion matrix
    D::SparseMatrixCSC{Float64,Int64}

    # inversion LHSs
    inversion_LHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}

    # U = 1 solution
    χ_U::Array{Float64,2}
end

################################################################################
# Constructors for ModelSetup2DPG
################################################################################

"""
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

Construct a ModelSetup2DPG struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup2DPG(bl::Bool, f::Float64, no_net_transport::Bool, L::Float64, nξ::Int64, nσ::Int64, coords::String, 
                    periodic::Bool, ξ::Array{Float64,1}, σ::Array{Float64,1}, H_func::Function, Hx_func::Function, 
                    ν_func::Function, κ_func::Function, N2_func::Function, Δt::Real)
    # evaluate functions 
    H = @. H_func(ξ)
    Hx = @. Hx_func(ξ)
    ν = zeros(nξ, nσ)
    κ = zeros(nξ, nσ)
    N2 = zeros(nξ, nσ)
    for i=1:nξ
        ν[i, :] = @. ν_func(ξ[i], σ)
        κ[i, :] = @. κ_func(ξ[i], σ)
        N2[i, :] = @. N2_func(ξ[i], σ)
    end

    # 2D coordinates in (x, z)
    x = repeat(ξ, 1, nσ)
    z = repeat(σ', nξ, 1).*repeat(H, 1, nσ)

    # pass to setup for arrays
    return ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt)
end

"""
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2 Δt)

Construct a ModelSetup2DPG struct using arrays of H, Hx, ν, and κ.
"""
function ModelSetup2DPG(bl::Bool, f::Float64, no_net_transport::Bool, L::Float64, nξ::Int64, nσ::Int64, coords::String, 
                    periodic::Bool, ξ::Array{Float64,1}, σ::Array{Float64,1}, x::Array{Float64,2}, z::Array{Float64,2}, 
                    H::Array{Float64,1}, Hx::Array{Float64,1}, ν::Array{Float64,2}, κ::Array{Float64,2}, N2::Array{Float64,2}, Δt::Real)
    # get derivative matrices
    Dξ = get_Dξ(ξ, L, periodic)
    Dσ = get_Dσ(σ)

    # get diffusion matrix
    D = get_D(ξ, σ, κ, H)

    # inversion LHSs
    inversion_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, nξ) 
    for i=1:nξ 
        inversion_LHSs[i] = get_inversion_LHS(ν[i, :], f, H[i], σ)
    end  
    
    # U = 1 inversion solution  
    inversion_RHS = get_inversion_RHS(f^2 ./ν, 1)
    χ_U = get_χ(inversion_LHSs, inversion_RHS) 

    return ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt, Dξ, Dσ, D, inversion_LHSs, χ_U)
end