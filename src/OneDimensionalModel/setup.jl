################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

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
    # use BL model or full?
    bl::Bool 

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
    inversion_LHS::SuiteSparse.UMFPACK.UmfpackLU

    # diffusion matrix
    D::SparseMatrixCSC{Float64,Int64}

    # transport constraint (boolean)
    transport_constraint::Bool

    # imposed transport (if transportConstraint == true)
    U::Vector{Float64}
end

################################################################################
# Constructors for ModelSetup1DPG
################################################################################

"""
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transportConstraint, U, Uamp, Uper)

Construct a ModelSetup1DPG struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1DPG(bl::Bool, f::Float64, nz::Int64, z::Vector{Float64}, H::Float64, θ::Float64, 
                    ν_func::Function, κ_func::Function, κ_z_func::Function,
                    N2::Float64, Δt::Real, transportConstraint::Bool, U::Vector{Float64})
    # evaluate functions 
    ν = ν_func.(z)
    κ = κ_func.(z)
    κ_z = κ_z_func.(z)

    # pass to next funciton below
    return ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U)
end

"""
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transportConstraint, U)

Construct a ModelSetup1DPG struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1DPG(bl::Bool, f::Float64, nz::Int64, z::Vector{Float64}, H::Float64, θ::Float64, 
                    ν::Vector{Float64}, κ::Vector{Float64}, κ_z::Vector{Float64},
                    N2::Float64, Δt::Real, transportConstraint::Bool, U::Vector{Float64})
    # inversion LHS
    inversionLHS = get_inversion_LHS(ν, z, f, transportConstraint) 

    # diffusion matrix
    D = get_D(z, κ)

    # return struct
    return ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, inversionLHS, D, transportConstraint, U)
end
