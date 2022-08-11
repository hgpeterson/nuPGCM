"""
    M = get_M(p, t, C₀)

Compute CSC sparse mass matrix `M` where M_ij = ∫ φᵢ φⱼ.
"""
function get_M(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # indices
	np = size(p, 1)
	nt = size(t, 1)

    # number of shape functions per triangle
    n = size(t, 2)

	# create global linear system using stamping method
    M = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to M from element k
        Mᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η)*shape_func(C₀[k, i, :], ξ, η)
                Mᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

		# add to global system
		for i=1:n
			for j=1:n
                push!(M, (t[k, i], t[k, j], Mᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    M = sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), np, np)

    return M
end

"""
    Cξ, Cη = get_Cξ_Cη(p, t, C₀)

Compute CSC sparse matrices `Cξ` and `Cη` where 
Cξ_ij = ∫ φᵢ ∂ξ(φⱼ) and Cη_ij = ∫ φᵢ ∂η(φⱼ).
"""
function get_Cξ_Cη(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # indices
	np = size(p, 1)
	nt = size(t, 1)

    # number of shape functions per triangle
    n = size(t, 2)

	# create global linear system using stamping method
    Cξ = Tuple{Int64,Int64,Float64}[]  
    Cη = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to Cξ from element k
        Cξᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η)
                Cξᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

		# calculate contribution to Cη from element k
        Cηᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η)
                Cηᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

		# add to global system
		for i=1:n
			for j=1:n
                push!(Cξ, (t[k, i], t[k, j], Cξᵏ[i, j]))
                push!(Cη, (t[k, i], t[k, j], Cηᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    Cξ = sparse((x->x[1]).(Cξ), (x->x[2]).(Cξ), (x->x[3]).(Cξ), np, np)
    Cη = sparse((x->x[1]).(Cη), (x->x[2]).(Cη), (x->x[3]).(Cη), np, np)

    return Cξ, Cη
end

"""
    CCξ, CCη = get_CCξ_CCη(m)

Compute CCξᵢⱼₖ = ∫ ∂ξ(φₖ) φⱼ φᵢ and CCηᵢⱼₖ = ∫ ∂η(φₖ) φⱼ φᵢ. 
"""
function get_CCξ_CCη(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    nt = size(t, 1)
    n = size(t, 2)
    o = convert(Int64, (-3 + sqrt(1 + 8*n))/2) # order of method
    d = 3*o - 1 # degree of integration
    CCξ = zeros(nt, n, n, n)
    CCη = zeros(nt, n, n, n)
    @showprogress "Computing CCξ and CCη..." for k₀=1:nt
        for i=1:n
            for j=1:n
                for k=1:n
                    func_ξ(x, y) = shape_func(C₀[k₀, k, :], x, y; dξ=1)*shape_func(C₀[k₀, j, :], x, y)*shape_func(C₀[k₀, i, :], x, y)
                    CCξ[k₀, i, j, k] = tri_quad(func_ξ, p[t[k₀, 1:3], :]; degree=d)

                    func_η(x, y) = shape_func(C₀[k₀, k, :], x, y; dη=1)*shape_func(C₀[k₀, j, :], x, y)*shape_func(C₀[k₀, i, :], x, y)
                    CCη[k₀, i, j, k] = tri_quad(func_η, p[t[k₀, 1:3], :]; degree=d)
                end
            end
        end
    end
    return CCξ, CCη
end

# function get_Ds(κ::AbstractArray{<:Real,2}, σ::AbstractArray{<:Real,1}, Δt)
#     # indices
#     nσ = size(σ, 1)

#     # interior nodes
#     D_LHS = Tuple{Int64,Int64,Float64}[]
#     D_RHS = Tuple{Int64,Int64,Float64}[]
#     for j=2:nσ-1
#         fσσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
#         push!(D_RHS, (j, j-1, fσσ[1]))
#         push!(D_RHS, (j, j,   fσσ[2]))
#         push!(D_RHS, (j, j+1, fσσ[3]))
#         push!(D_LHS, (j, j-1, 1 - Δt/2*fσσ[1]))
#         push!(D_LHS, (j, j,   1 - Δt/2*fσσ[2]))
#         push!(D_LHS, (j, j+1, 1 - Δt/2*fσσ[3]))
#     end

#     # flux boundary conditions at σ = -1, 0
#     fσ_bot = mkfdstencil(σ[1:3], σ[1], 1)
#     push!(D_LHS, (1, 1, fσ_bot[1]))
#     push!(D_LHS, (1, 2, fσ_bot[2]))
#     push!(D_LHS, (1, 3, fσ_bot[3]))
#     fσ_top = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
#     push!(D_LHS, (nσ, nσ-2, fσ_top[1]))
#     push!(D_LHS, (nσ, nσ-1, fσ_top[2]))
#     push!(D_LHS, (nσ, nσ,   fσ_top[3]))

#     D_LHS = sparse((x->x[1]).(D_LHS), (x->x[2]).(D_LHS), (x->x[3]).(D_LHS), nσ, nσ)
#     D_RHS = sparse((x->x[1]).(D_RHS), (x->x[2]).(D_RHS), (x->x[3]).(D_RHS), nσ, nσ)
#     return lu(D_LHS), D_RHS
# end
function get_D_LHS(κ::AbstractArray{<:Real,2}, σ::AbstractArray{<:Real,1}, H::AbstractArray{<:Real,1}, Δt::Real)
    # indices
    np = size(H, 1)
    nσ = size(σ, 1)
    imap = reshape(1:np*nσ, np, nσ)

    # interior nodes
    D = Tuple{Int64,Int64,Float64}[]
    for i=1:np
        # interior diffusion
        for j=2:nσ-1
            fσ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            κσ = dot(fσ, κ[i, j-1:j+1])
            fσσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
            # (κ_σ*b_σ + κ*b_σσ)/H^2
            push!(D, (imap[i, j], imap[i, j-1], 1 - Δt/2*(κσ*fσ[1] + κ[i, j]*fσσ[1])/H[i]^2))
            push!(D, (imap[i, j], imap[i, j],   1 - Δt/2*(κσ*fσ[2] + κ[i, j]*fσσ[2])/H[i]^2))
            push!(D, (imap[i, j], imap[i, j+1], 1 - Δt/2*(κσ*fσ[3] + κ[i, j]*fσσ[3])/H[i]^2))
        end

        # flux boundary conditions at σ = -1, 0
        fσ_bot = mkfdstencil(σ[1:3], σ[1], 1)
        push!(D, (imap[i, 1], imap[i, 1], fσ_bot[1]/H[i]))
        push!(D, (imap[i, 1], imap[i, 2], fσ_bot[2]/H[i]))
        push!(D, (imap[i, 1], imap[i, 3], fσ_bot[3]/H[i]))
        fσ_top = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        push!(D, (imap[i, nσ], imap[i, nσ-2], fσ_top[1]/H[i]))
        push!(D, (imap[i, nσ], imap[i, nσ-1], fσ_top[2]/H[i]))
        push!(D, (imap[i, nσ], imap[i, nσ],   fσ_top[3]/H[i]))
    end

    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), np*nσ, np*nσ)
    return D
end
function get_Dσ(σ::AbstractArray{<:Real,1})
    nσ = size(σ, 1)
    Dσσ = Tuple{Int64,Int64,Float64}[]
    for j=2:nσ-1
        fσ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
        push!(Dσσ, (j, j-1, fσσ[1]))
        push!(Dσσ, (j, j,   fσσ[2]))
        push!(Dσσ, (j, j+1, fσσ[3]))
    end
    fσ = mkfdstencil(σ[1:3], σ[1], 1)
    push!(Dσσ, (1, 1, fσ[1]))
    push!(Dσσ, (1, 2, fσ[2]))
    push!(Dσσ, (1, 3, fσ[3]))
    fσ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    push!(Dσσ, (nσ, nσ-2, fσ[1]))
    push!(Dσσ, (nσ, nσ-1, fσ[2]))
    push!(Dσσ, (nσ, nσ,   fσ[3]))
    return sparse((x->x[1]).(Dσ), (x->x[2]).(Dσ), (x->x[3]).(Dσ), nσ, nσ)
end
function get_Dσσ(σ::AbstractArray{<:Real,1})
    nσ = size(σ, 1)
    Dσσ = Tuple{Int64,Int64,Float64}[]
    for j=2:nσ-1
        fσσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        push!(Dσσ, (j, j-1, fσσ[1]))
        push!(Dσσ, (j, j,   fσσ[2]))
        push!(Dσσ, (j, j+1, fσσ[3]))
    end
    push!(Dσσ, (1, 1, 1))
    push!(Dσσ, (nσ, nσ, 1))
    Dσσ = sparse((x->x[1]).(Dσσ), (x->x[2]).(Dσσ), (x->x[3]).(Dσσ), nσ, nσ)
end

"""
    v₀ = fem_evaluate(m, v, ξ, η)
    v₀ = fem_evaluate(m, v, ξ, η, k)

Define FEM evaluation function with ModelSetup3DPG struct (see ../Numerics/finite_elements.jl
for original definition).
"""
function fem_evaluate(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ::Real, η::Real)
    return fem_evaluate(v, ξ, η, m.p, m.t, m.t_dict, m.C₀)
end
function fem_evaluate(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ::Real, η::Real, k::Integer)
    return fem_evaluate(v, ξ, η, m.p, m.t, m.C₀, k)
end

"""
    ∂ᵢv₀ = fem_evaluate(v, ξ, η, p, t, t_dict, C₀, i)
    ∂ᵢv₀ = fem_evaluate(v, ξ, η, k, p, t, C₀, i)
    ∂ᵢv₀ = fem_evaluate(m, v, ξ, η, i)
    ∂ᵢv₀ = fem_evaluate(m, v, ξ, η, k, i)

Evaluate derivative of `v` in `i` direction at (ξ, η).
"""
function ∂ᵢ(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, t_dict::AbstractDict{IN, Vector{IN}}, C₀::AbstractArray{<:Real,3}; 
            i::Integer) where IN <: Integer
    # find triangle p₀ is in
    k = get_tri(ξ, η, p, t, t_dict)

    # evaluate there
    return ∂ᵢ(v, ξ, η, k, p, t, C₀; i)
end
function ∂ᵢ(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, k::Integer, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # sum weighted combinations of shape function derivatives
    n = size(t, 2)
    ∂v = 0
    for j=1:n
        ∂v += v[t[k, j]]*shape_func(C₀[k, j, :], ξ, η; dξ=Int(i==1), dη=Int(i==2))
    end
    return ∂v
end
function ∂ᵢ(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ::Real, η::Real; i::Integer)
    return ∂ᵢ(v, ξ, η, m.p, m.t, m.t_dict, m.C₀; i)
end
function ∂ᵢ(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ::Real, η::Real, k::Integer; i::Integer)
    return ∂ᵢ(v, ξ, η, k, m.p, m.t, m.C₀; i)
end

"""
Explicitly define ξ and η derivatives in terms of ∂ᵢ.
"""
function ∂ξ(args...)
    return ∂ᵢ(args...; i=1)
end
function ∂η(args...)
    return ∂ᵢ(args...; i=2)
end