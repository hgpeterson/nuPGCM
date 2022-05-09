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

	# Coriolis parameter (s⁻¹)
    f::AbstractArray{FT,1}

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

    # wind stress
    τ_wξ::AbstractArray{FT,3}

    # transport stress
    τ_tξ::AbstractArray{FT,3}
end

################################################################################
# Constructors for ModelSetup3DPG
################################################################################

function ModelSetup3DPG(bl, ρ₀, f, fy, Lx, Ly, p, t, e, σ, H, Hx, Hy, ν, κ, N², Δt)
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
    @inbounds @showprogress "Computing baroclinic components..." for i=1:np 
        if i in e
            baroclinic_LHSs[i] = nothing
        else
            baroclinic_LHSs[i] = get_baroclinic_LHS(ρ₀, ν[i, :], f[i], H[i], σ)
        end
    end  

    # τ's
    baroclinic_RHSs_wξ = zeros(np, 2*nσ)
    baroclinic_RHSs_tξ = zeros(np, 2*nσ)
    @inbounds for i=1:np
        if i in e
            continue
        else
            baroclinic_RHSs_wξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 1, 0, 0, 0)
            baroclinic_RHSs_tξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 0)
        end
    end
    τ_tξ = get_τ(baroclinic_LHSs, baroclinic_RHSs_tξ)
    τ_wξ = get_τ(baroclinic_LHSs, baroclinic_RHSs_wξ)

    # compute barotropic LHS matrix
    barotropic_LHS = get_barotropic_LHS(p, t, e, C₀, ρ₀, f, fy, H, Hx, Hy, τ_tξ)

    println("Setup complete!\n")

    return ModelSetup3DPG(bl, ρ₀, f, Lx, Ly, np, nt, ne, nσ, p, t, e, C₀, σ, H, Hx, Hy, ν, κ, N², Δt, baroclinic_LHSs, barotropic_LHS, τ_tξ, τ_wξ)
end