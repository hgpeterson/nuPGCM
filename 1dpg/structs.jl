################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

using SparseArrays, SuiteSparse, LinearAlgebra

struct ModelState1DPG
    # buoyancy (m s-2)
	b::Vector{Float64}

    # streamfunction (m2 s-1)
    χ::Vector{Float64}

    # velocities (m s-1)
	u::Vector{Float64}
	v::Vector{Float64}

    # iteration
    i::Vector{Int64}
end

struct ModelSetup1DPG
	# Coriolis parameter (s-1)
	f::Float64

	# number of grid points
	nz::Int64

	# grid coordinates
	z::Vector{Float64}

    # depth (m)
    H::Float64

    # slope angle (rad)
    θ::Float64

    # turbulent viscosity (m2 s-1)
	ν::Vector{Float64}

    # turbulent diffusivity (m2 s-1)
	κ::Vector{Float64}

    # derivative of κ (m s-1)
	κ_z::Vector{Float64}

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

    # imposed transport (if transportConstraint == true)
    U::Vector{Float64}

    # amplitude of tidal oscillations (can be zero for no tides)
    Uamp::Float64

    # period of tidal oscillations
    Uper::Float64
end
