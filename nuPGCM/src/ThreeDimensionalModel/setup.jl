################################################################################
# Model structs for
#   (1) Current State 
#   (2) Setup/Params
################################################################################

struct ModelState3DPG
    # buoyancy (m s-2)
	b::Array{Float64,2}
    
    # barotropic streamfunction
    Ψ::Array{Float64,1}

    # velocities (m s-1)
	uξ::Array{Float64,2}
	uη::Array{Float64,2}
	uσ::Array{Float64,2}

    # iteration
    i::Array{Int64,1}
end

struct ModelSetup3DPG
    # use BL model or full?
    bl::Bool 

	# Coriolis parameter (s-1)
	f::Float64

	# number of grid points
	n_nodes::Int64
	nσ::Int64

    # horizontal mesh points and triangles
    p::Array{Float64,2}
    t::Array{Float64,2}

	# vertical grid 
	σ::Array{Float64,1}

    # depth (m)
    H::Array{Float64, 1}

    # derivatives of depth w.r.t. x and y
    Hx::Array{Float64,1}
    Hy::Array{Float64,1}

    # turbulent viscosity (m2 s-1)
	ν::Array{Float64,2}

    # turbulent diffusivity (m2 s-1)
	κ::Array{Float64,2}

    # buoyancy frequency (s-2)
	N2::Array{Float64,2}

    # timestep (s)
	Δt::Float64
end

################################################################################
# Constructors for ModelSetup3DPG
################################################################################