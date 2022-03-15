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