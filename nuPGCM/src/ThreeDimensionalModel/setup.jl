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
    baroclinic_LHSs::AbstractArray{Any}

    # barotropic LHS matrix
    barotropic_LHS::SuiteSparse.UMFPACK.UmfpackLU{FT,IT}

    # transport stress
    τξ_tξ::AbstractArray{FT,2}
    τη_tη::AbstractArray{FT,2}
    τξ_tη::AbstractArray{FT,2}
    τη_tξ::AbstractArray{FT,2}
end

################################################################################
# Constructors for ModelSetup3DPG
################################################################################

function ModelSetup3DPG(bl, f_func, fy_func, p, t, e, σ, H_func, Hx_func, Hy_func, ν, κ, N², Δt)
    np = size(p, 1)
    nσ = size(σ, 1)

    # evaluate functions
    ξ = p[:, 1]
    η = p[:, 2]
    f = f_func.(ξ, η)
    H = H_func.(ξ, η)
    Hx = Hx_func.(ξ, η)
    Hy = Hy_func.(ξ, η)

    # compute shape function coefficients
    C₀ = get_linear_basis_coeffs(p, t)

    # compute baroclinic LHS matrices
    println("computing baroclinic components")
    baroclinic_LHSs = Array{Any}(undef, np) 
    @inbounds for i=1:np 
        if i in e
            baroclinic_LHSs[i] = nothing
        else
            baroclinic_LHSs[i] = get_baroclinic_LHS(ν[i, :], f[i], H[i], σ)
        end
    end  

    # compute τ_t
    baroclinic_RHSs_tξ = zeros(np, 2*nσ)
    baroclinic_RHSs_tη = zeros(np, 2*nσ)
    @inbounds for i=1:np
        if i in e
            continue
        else
            baroclinic_RHSs_tξ[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 0)
            baroclinic_RHSs_tη[i, :] = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 0, 1)
        end
    end
    τξ_tξ, τη_tξ = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs_tξ)
    τξ_tη, τη_tη = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs_tη)

    # compute barotropic LHS matrix
    # r = 5e-6
    # τξ_tξ_bot_func(ξ, η) = r
    # τη_tη_bot_func(ξ, η) = r
    # τξ_tη_bot_func(ξ, η) = r/1e2
    # τη_tξ_bot_func(ξ, η) = r/1e2
    τξ_tξ_bot_func(ξ, η) = evaluate(τξ_tξ[:, 1], [ξ, η], p, t, C₀)
    τη_tη_bot_func(ξ, η) = evaluate(τη_tη[:, 1], [ξ, η], p, t, C₀)
    τξ_tη_bot_func(ξ, η) = evaluate(τξ_tη[:, 1], [ξ, η], p, t, C₀)
    τη_tξ_bot_func(ξ, η) = evaluate(τη_tξ[:, 1], [ξ, η], p, t, C₀)
    barotropic_LHS = get_barotropic_LHS(p, t, e, C₀, f_func, fy_func, H_func, Hx_func, Hy_func, τξ_tξ_bot_func, τη_tη_bot_func, τξ_tη_bot_func, τη_tξ_bot_func)

    return ModelSetup3DPG(bl, f, np, nσ, p, t, e, C₀, σ, H, Hx, Hy, ν, κ, N², Δt, baroclinic_LHSs, barotropic_LHS, τξ_tξ, τη_tη, τξ_tη, τη_tξ)
end
