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

    # reference density (kg m⁻³)
    ρ₀::FT

	# Coriolis parameter f = f₀ + βη (s⁻¹)
    f₀::FT
    β::FT

    # basin widths (m)
    Lx::FT
    Ly::FT

	# number of mesh points
	np::IT

	# number of triangles
	nt::IT

	# number of edge points
	ne::IT

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
    baroclinic_LHSs::AbstractArray{Any}

    # barotropic LHS matrix
    barotropic_LHS::SuiteSparse.UMFPACK.UmfpackLU{FT,IT}

    # transport stress
    τ_tξ::AbstractArray{FT,3}

    # wind stress
    τ_wξ::AbstractArray{FT,3}
end

################################################################################
# Constructors for ModelSetup3DPG
################################################################################

function ModelSetup3DPG(bl, ρ₀, f₀, β, Lx, Ly, p, t, e, σ, H, Hx, Hy, ν, κ, N², Δt)
    # indices
    np = size(p, 1)
    nt = size(t, 1)
    ne = size(e, 1)
    nσ = size(σ, 1)

    # coords
    ξ = p[:, 1]
    η = p[:, 2]

    # shape function coefficients
    C₀ = get_linear_basis_coeffs(p, t)

    # baroclinic LHS matrices
    baroclinic_LHSs = Array{Any}(undef, np) 
    @inbounds @showprogress "Computing baroclinic_LHSs..." for i=1:np 
        # if i in e
        #     baroclinic_LHSs[i] = nothing
        # else
        #     baroclinic_LHSs[i] = get_baroclinic_LHS(ρ₀, ν[i, :], f[i], H[i], σ)
        # end
        baroclinic_LHSs[i] = get_baroclinic_LHS(ρ₀, ν[i, :], f₀ + β*η[i], H[i], σ)
    end  

    # compute m = ∫ φᵢ 
    m = get_m(p, t, C₀)

    # compute M = ∫ φᵢ φⱼ
    M = get_M(p, t, e, C₀)

    # baroclinic RHSs for wind and transport terms
    baroclinic_RHSs_tξ = zeros(np, 2*nσ)
    baroclinic_RHSs_wξ = zeros(np, 2*nσ)
    @inbounds for i=1:np
        # if i in e
        #     continue
        # else
        #     baroclinic_RHSs_tξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, m[i], 0)
        #     baroclinic_RHSs_wξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), m[i], 0, 0, 0) 
        # end
        baroclinic_RHSs_wξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), m[i], 0,    0, 0) 
        baroclinic_RHSs_tξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ),    0, 0, m[i], 0)
    end

    # solve for v = M τ
    v_tξ = get_v(baroclinic_LHSs, baroclinic_RHSs_tξ)
    v_wξ = get_v(baroclinic_LHSs, baroclinic_RHSs_wξ)

    # invert for τ
    τ_tξ = zeros(2, np, nσ)
    τ_tξ[1, :, :] = M\v_tξ[1, :, :]
    τ_tξ[2, :, :] = M\v_tξ[2, :, :]
    τ_wξ = zeros(2, np, nσ)
    τ_wξ[1, :, :] = M\v_wξ[1, :, :]
    τ_wξ[2, :, :] = M\v_wξ[2, :, :]

    # compute barotropic LHS matrix
    barotropic_LHS = get_barotropic_LHS(p, t, e, C₀, ρ₀, f₀, β, H, Hx, Hy, τ_tξ)

    println("Setup complete!\n")

    return ModelSetup3DPG(bl, ρ₀, f₀, β, Lx, Ly, np, nt, ne, nσ, p, t, e, C₀, 
                          σ, H, Hx, Hy, ν, κ, N², Δt, baroclinic_LHSs, barotropic_LHS, 
                          τ_tξ, τ_wξ)
end