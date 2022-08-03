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
    # p::AbstractArray{FT,2}
    p::Array{FT,2}

    # mesh triangles
    # t::AbstractArray{IT,2}
    t::Array{IT,2}

    # mesh outer edges
    # e::AbstractArray{IT,1}
    e::Array{IT,1}

    # dictionary of triangles each point is in
    t_dict::Dict{IT, Vector{IT}}

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
    baroclinic_LHSs::AbstractArray{SuiteSparse.UMFPACK.UmfpackLU}

    # barotropic LHS matrix
    barotropic_LHS::SuiteSparse.UMFPACK.UmfpackLU{FT,IT}

    # mass matrix and its LU decomposition
    M::SparseArrays.SparseMatrixCSC{FT, IT}
    M_LU::SuiteSparse.UMFPACK.UmfpackLU{FT,IT}

    # derivative matrices
    Cξ::SparseArrays.SparseMatrixCSC{FT, IT}
    Cη::SparseArrays.SparseMatrixCSC{FT, IT}
    CCξ::AbstractArray{FT,4}
    CCη::AbstractArray{FT,4}

    # transport stress
    τξ_tξ::AbstractArray{FT,2}
    τη_tξ::AbstractArray{FT,2}

    # wind stress
    τξ_wξ::AbstractArray{FT,2}
    τη_wξ::AbstractArray{FT,2}
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

    # create dictionary of triangles each point is in
    t_dict = get_t_dict(p, t)

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)

    # baroclinic LHS matrices
    baroclinic_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU}(undef, np) 
    @showprogress "Computing baroclinic_LHSs..." for i=1:np 
        baroclinic_LHSs[i] = get_baroclinic_LHS(ρ₀, ν[i, :], f₀ + β*η[i], H[i], σ)
    end  

    # compute m = ∫ φᵢ 
    m = get_m(p, t, C₀)

    # compute M = ∫ φᵢ φⱼ
    M = get_M(p, t, C₀)
    M_LU = lu(M)

    # compute Cξ = ∫ φᵢ ∂ξ(φⱼ), Cη = ∫ φᵢ ∂η(φⱼ)
    Cξ, Cη = get_Cξ_Cη(p, t, C₀)

    # compute CCξᵢⱼₖ = ∫ ∂ξ(φₖ) φⱼ φᵢ and CCηᵢⱼₖ = ∫ ∂η(φₖ) φⱼ φᵢ. 
    CCξ, CCη = get_CCξ_CCη(p, t, C₀)

    # baroclinic RHSs for wind and transport terms
    baroclinic_RHSs_tξ = zeros(np, 2*nσ)
    baroclinic_RHSs_wξ = zeros(np, 2*nσ)
    for i=1:np
        baroclinic_RHSs_tξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 0)
        baroclinic_RHSs_wξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 1, 0, 0, 0) 
    end

    # solve for τ at each column
    τξ_tξ, τη_tξ = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs_tξ)
    τξ_wξ, τη_wξ = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs_wξ)

    # compute barotropic LHS matrix
    barotropic_LHS = get_barotropic_LHS(p, t, e, C₀, ρ₀, f₀, β, H, Hx, Hy, τξ_tξ[:, 1], τη_tξ[:, 1])

    println("Setup complete!\n")

    return ModelSetup3DPG(bl, ρ₀, f₀, β, Lx, Ly, np, nt, ne, nσ, p, t, e, t_dict, C₀, 
                          σ, H, Hx, Hy, ν, κ, N², Δt, baroclinic_LHSs, barotropic_LHS, 
                          M, M_LU, Cξ, Cη, CCξ, CCη, τξ_tξ, τη_tξ, τξ_wξ, τη_wξ)
end