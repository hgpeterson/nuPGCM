################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState2D{M<:AbstractMatrix, V<:AbstractVector}
    # buoyancy (m s-2)
	b::M

    # streamfunction (m2 s-1)
    χ::M

    # velocities (m s-1)
	uξ::M
	uη::M
	uσ::M

    # iteration
    i::V
end

struct ModelSetup2D{I<:Integer, F<:AbstractFloat, V<:AbstractVector, M<:AbstractMatrix, SM<:SparseMatrixCSC, SV<:AbstractVector}
    # use BL model or full?
    bl::Bool 

	# Coriolis parameter (s-1)
	f::F

	# U = 0 or no?
	no_net_transport::Bool

    # width of domain (m)
	L::F

	# number of grid points
	nξ::I
	nσ::I

    # coordinates
    coords::String

    # periodic in x direction?
    periodic::Bool

	# grid coordinates
	ξ::V
	σ::V
    x::M
    z::M

    # depth (m)
    H::V

    # derivative of depth w.r.t. x
    Hx::V

    # turbulent viscosity (m2 s-1)
	ν::M

    # turbulent diffusivity (m2 s-1)
	κ::M

    # buoyancy frequency (s-2)
	N2::M

    # timestep (s)
	Δt::F

    # derivative matrices
    Dξ::SM
    Dσ::SM

    # diffusion matrix
    D::SM

    # inversion LHSs
    inversion_LHSs::SV

    # U = 1 solution
    χ_U::M
end

################################################################################
# Constructors for ModelSetup2DPG
################################################################################

"""
    m = ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

Construct a ModelSetup2D struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, 
                      H_func::Function, Hx_func::Function, ν_func::Function, κ_func::Function, N2_func::Function, Δt)
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
    return ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt)
end

"""
    m = ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2 Δt)

Construct a ModelSetup2D struct using arrays of H, Hx, ν, and κ.
"""
function ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, 
                      H::V, Hx::V, ν::M, κ::M, N2::M, Δt) where {V<:AbstractVector, M<:AbstractMatrix}
    # get derivative matrices
    Dξ = get_Dξ(ξ, L, periodic)
    Dσ = get_Dσ(σ)

    # get diffusion matrix
    D = get_D(ξ, σ, κ, H)

    # inversion LHSs
    inversion_LHSs = [get_inversion_LHS(ν[i, :], f, H[i], σ) for i=1:nξ]
    
    # U = 1 inversion solution  
    inversion_RHS = get_inversion_RHS(f^2 ./ν, 1)
    χ_U = get_χ(inversion_LHSs, inversion_RHS) 

    return ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt, Dξ, Dσ, D, inversion_LHSs, χ_U)
end