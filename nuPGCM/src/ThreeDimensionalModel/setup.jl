################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState3DPG{FT,IT}
    # buoyancy (m s⁻²)
	b::AbstractArray{FT,2}
    
    # barotropic streamfunction
    Ψ::AbstractArray{FT,1}

    # velocities (m s⁻¹)
	uξ::AbstractArray{FT,2}
	uη::AbstractArray{FT,2}
	uσ::AbstractArray{FT,2}

    # iteration
    i::AbstractArray{IT,1}
end

struct ModelSetup3DPG{FT,IT}
    # use BL model or full?
    bl::Bool 

	# Coriolis parameter (s⁻¹)
    f::AbstractArray{FT,1}

	# number of mesh points
	np::IT

    # number of vertical grid points
	nσ::IT

    # mesh points
    p::AbstractArray{FT,2}

    # mesh triangles
    t::AbstractArray{IT,2}

    # mesh outter edges
    e::AbstractArray{IT,1}

    # shape function coefficients
    C₀::AbstractArray{FT,3}

	# vertical grid 
	σ::AbstractArray{FT,1}

    # depth (m)
    H::AbstractArray{FT,1}

    # ∂H/∂x and ∂H/∂y
    Hx::AbstractArray{FT,1}
    Hy::AbstractArray{FT,1}

    # turbulent viscosity and diffusivity (m² s⁻¹)
	ν::AbstractArray{FT,2}
	κ::AbstractArray{FT,2}

    # buoyancy frequency (s⁻²)
	N²::AbstractArray{FT,2}

    # timestep (s)
	Δt::FT

    # baroclinic LHS matrices
    baroclinic_LHSs::AbstractArray{SuiteSparse.UMFPACK.UmfpackLU{FT,IT}}

    # barotropic LHS matrix
    barotropic_LHS::SuiteSparse.UMFPACK.UmfpackLU{FT,IT}

    # transport stress
    τξ_t::AbstractArray{FT,2}
    τη_t::AbstractArray{FT,2}
end

################################################################################
# Constructors for ModelSetup3DPG
################################################################################

function ModelSetup3DPG(bl, f, p, t, e, σ, H, Hx, Hy, ν, κ, N², Δt)
    np = size(p, 1)
    nσ = size(σ, 1)

    # compute shape function coefficients
    C₀ = get_linear_basis_coeffs(p, t)

    # compute baroclinic LHS matrices
    baroclinic_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, np) 
    @inbounds for i=1:np 
        baroclinic_LHSs[i] = get_baroclinic_LHS(ν[i, :], f[i], H[i], σ)
    end  

    # compute τ_t
    baroclinic_RHSs = zeros(np, 2*nσ)
    @inbounds for i=1:np
        baroclinic_RHSs[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 1)
    end
    τξ_t, τη_t = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs)

    # compute barotropic LHS matrix
    # barotropic_LHS = get_barotropic_LHS(p, t, e, f, H, Hx, Hy, τξ_t[:, 1], τη_t[:, 1])
    barotropic_LHS = baroclinic_LHSs[1]

    return ModelSetup3DPG(bl, f, np, nσ, p, t, e, C₀, σ, H, Hx, Hy, ν, κ, N², Δt, baroclinic_LHSs, barotropic_LHS, τξ_t, τη_t)
end
