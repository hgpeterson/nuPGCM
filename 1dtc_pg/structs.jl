################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

using SparseArrays, SuiteSparse, LinearAlgebra

struct ModelState1DPG
    # buoyancy (m s-2)
	b::Array{Float64,1}

    # streamfunction (m2 s-1)
    χ::Array{Float64,1}

    # velocities (m s-1)
	u::Array{Float64,1}
	v::Array{Float64,1}

    # transport (m2 s-1)
    U::Array{Float64,1}

    # iteration
    i::Array{Int64,1}
end

struct ModelSetup1DPG
	# Coriolis parameter (s-1)
	f::Float64

	# number of grid points
	nz::Int64

	# grid coordinates
	z::Array{Float64,1}

    # depth (m)
    H::Float64

    # slope angle (rad)
    θ::Float64

    # turbulent viscosity (m2 s-1)
	ν::Array{Float64,1}

    # turbulent diffusivity (m2 s-1)
	κ::Array{Float64,1}

    # derivative of κ (m s-1)
	κ_z::Array{Float64,1}

    # buoyancy frequency (s-2)
	N2::Float64

    # timestep (s)
	Δt::Float64

    # inversion LHS
    inversionLHS::SuiteSparse.UMFPACK.UmfpackLU

    # diffusion matrix
    D::SparseMatrixCSC{Float64,Int64}

    # transport constraint (boolean)
    transportConstraint::Bool

    # transport
    U₀::Float64
end
