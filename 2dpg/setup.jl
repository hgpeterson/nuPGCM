################################################################################
# Model setup
################################################################################

include("structs.jl")
include("utils.jl")
include("inversion.jl")
include("evolution.jl")

"""
    m = ModelSetup(f, ξVariation, L, nξ, nσ, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

Construct a ModelSetup struct using analytical functions of H, Hx, ν, κ, and N.
"""
function ModelSetup2DPG(f::Float64, ξVariation::Bool, L::Float64, nξ::Int64, nσ::Int64, coords::String, 
                    periodic::Bool, ξ::Array{Float64,1}, σ::Array{Float64,1}, H_func::Function, Hx_func::Function, 
                    ν_func::Function, κ_func::Function, N2_func::Function, Δt::Real)
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
    return ModelSetup2DPG(f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt)
end

"""
    m = ModelSetup(f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2 Δt)

Construct a ModelSetup struct using arrays of H, Hx, ν, and κ.
"""
function ModelSetup2DPG(f::Float64, ξVariation::Bool, L::Float64, nξ::Int64, nσ::Int64, coords::String, 
                    periodic::Bool, ξ::Array{Float64,1}, σ::Array{Float64,1}, x::Array{Float64,2}, z::Array{Float64,2}, 
                    H::Array{Float64,1}, Hx::Array{Float64,1}, ν::Array{Float64,2}, κ::Array{Float64,2}, N2::Array{Float64,2}, Δt::Real)
    # get derivative matrices
    Dξ = getDξ(ξ, L, periodic)
    Dσ = getDσ(σ)

    # get diffusion matrix
    D = getDiffusionMatrix(ξ, σ, κ, H)

    # inversion LHSs
    inversionLHSs = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}}(undef, nξ) 
    for i=1:nξ 
        inversionLHSs[i] = getInversionLHS(ν[i, :], f, H[i], σ)
    end  
    
    # U = 1 inversion solution  
    inversionRHS = getInversionRHS(f^2 ./ν, 1)
    χ_U = computeχ(inversionLHSs, inversionRHS) 

    return ModelSetup2DPG(f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt, Dξ, Dσ, D, inversionLHSs, χ_U)
end