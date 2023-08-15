################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState1D{IV<:AbstractVector, FV<:AbstractVector}
    # buoyancy (m s-2)
	b::FV

    # streamfunction (m2 s-1)
    χ::FV

    # velocities (m s-1)
	u::FV
	v::FV

    # iteration
    i::IV
end

struct ModelSetup1D{FT<:AbstractFloat, IN<:Integer, V<:AbstractVector, FA<:Factorization, SM<:SparseMatrixCSC}
    # use BL model or full?
    bl::Bool 

	# Coriolis parameter (s-1)
	f::FT

	# number of grid points
	nz::IN

	# grid coordinates
	z::V

    # depth (m)
    H::FT

    # slope angle (rad)
    θ::FT

    # turbulent viscosity (m2 s-1)
	ν::V

    # turbulent diffusivity (m2 s-1)
	κ::V

    # derivative of κ (m s-1)
	κ_z::V

    # buoyancy frequency (s-2)
	N2::FT

    # timestep (s)
	Δt::FT

    # inversion LHS
    inversion_LHS::FA

    # diffusion matrix
    D::SM

    # transport constraint (boolean)
    transport_constraint::Bool

    # imposed transport (if transport_constraint == true)
    U::V
end

################################################################################
# Constructors for ModelSetup1DPG
################################################################################

"""
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transport_constraint, U)

Construct a ModelSetup1DPG struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1D(bl, f, nz, z, H, θ, ν_func::Function, κ_func::Function, κ_z_func::Function,
                      N2, Δt, transport_constraint, U)
    # evaluate functions 
    ν = ν_func.(z)
    κ = κ_func.(z)
    κ_z = κ_z_func.(z)

    # pass to next funciton below
    return ModelSetup1D(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transport_constraint, U)
end

"""
    m = ModelSetup1DPG(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transport_constraint, U)

Construct a ModelSetup1DPG struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup1D(bl, f, nz, z, H, θ, ν::V, κ::V, κ_z::V, N2, Δt, transport_constraint, U) where V <: AbstractVector
    # inversion LHS
    inversion_LHS = get_inversion_LHS(ν, f, z, transport_constraint) 

    # diffusion matrix
    D = get_D(z, κ)

    # return struct
    return ModelSetup1D(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, inversion_LHS, D, transport_constraint, U)
end
