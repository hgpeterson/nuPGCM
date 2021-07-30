immutable ModelParams
	# turbulent Prandtl number
	Pr::Float64

	# Coriolis parameter (s-1)
	f::Float64

	# buoyancy frequency
	N::Float64

	# turn on/off variations in ξ
	ξVariation::Bool

	# width of domain
	L::Float64

	# height of domain
	H0::Float64

	# number of grid points
	nξ::Int64
	nσ::Int64

	# grid arrays in terrain-following (ξ, σ) space
	σ::Array{Float64, 1}
	ξ::Array{Float64, 1}

	# grid arrays in physical (x, z) space
	x::Array{Float64, 2}
	z::Array{Float64, 2}

	# turbulent viscosity
	ν::Array{Float64, 2}

	# turbulent diffusivity
	κ::Array{Float64, 2}

	# timestep
	Δt::Float64

	function ModelParams()
		new()
	end
end

type ModelState
	b::Array{Float64, 2}

	i::Int64
end