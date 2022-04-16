################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState3DPG
    # buoyancy (m s⁻²)
	b::Array{Float64,2}
    
    # barotropic streamfunction
    Ψ::Array{Float64,1}

    # velocities (m s⁻¹)
	uξ::Array{Float64,2}
	uη::Array{Float64,2}
	uσ::Array{Float64,2}

    # iteration
    i::Array{Int64,1}
end

struct ModelSetup3DPG
    # use BL model or full?
    bl::Bool 

	# Coriolis parameter (s⁻¹)
    f::Array{Float64,1}

	# number of mesh points
	np::Int64

    # number of vertical grid points
	nσ::Int64

    # mesh points
    p::Array{Float64,2}

    # mesh triangles
    t::Array{Int64,2}

    # mesh outter edges
    e::Array{Int64,1}

    # shape function coefficients
    C₀::Array{Float64,3}

	# vertical grid 
	σ::Array{Float64,1}

    # depth (m)
    H::Array{Float64,1}

    # ∂H/∂x and ∂H/∂y
    Hx::Array{Float64,1}
    Hy::Array{Float64,1}

    # turbulent viscosity and diffusivity (m² s⁻¹)
	ν::Array{Float64,2}
	κ::Array{Float64,2}

    # buoyancy frequency (s⁻²)
	N²::Array{Float64,2}

    # timestep (s)
	Δt::Float64

    # baroclinic LHS matrices
    baroclinic_LHSs::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}

    # barotropic LHS matrix
    barotropic_LHS::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}

    # transport stress
    τξ_t::Array{Float64,2}
    τη_t::Array{Float64,2}
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
    for i=1:np 
        baroclinic_LHSs[i] = get_baroclinic_LHS(ν[i, :], f[i], H[i], σ)
    end  

    # compute τ_t
    baroclinic_RHSs = zeros(np, 2*nσ)
    for i=1:np
        baroclinic_RHSs[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 1)
    end
    τξ_t, τη_t = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs)

    # compute barotropic LHS matrix
    # barotropic_LHS = get_barotropic_LHS(p, t, e, f, H, Hx, Hy, τξ_t[:, 1], τη_t[:, 1])
    barotropic_LHS = baroclinic_LHSs[1]

    return ModelSetup3DPG(bl, f, np, nσ, p, t, e, C₀, σ, H, Hx, Hy, ν, κ, N², Δt, baroclinic_LHSs, barotropic_LHS, τξ_t, τη_t)
end
