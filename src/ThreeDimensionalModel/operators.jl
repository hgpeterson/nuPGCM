function get_M(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, e::AbstractArray{<:Integer,1}, C₀::AbstractArray{<:Real,3})
    # indices
	np = size(p, 1)
	nt = size(t, 1)
	ne = size(e, 1)

	# create global linear system using stamping method
    M = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to M from element k
        Mᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = local_basis_func(C₀[k, :, j], ξ, η)*local_basis_func(C₀[k, :, i], ξ, η)
                Mᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
                # if t[k, i] in e
                #     continue
                # else
                #     push!(M, (t[k, i], t[k, j], Mᵏ[i, j]))
                # end
                push!(M, (t[k, i], t[k, j], Mᵏ[i, j]))
			end
		end
	end
    # for i=1:ne
    #     push!(M, (e[i], e[i], 1))
    # end

    # make CSC matrix
    M = sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), np, np)

    return M
end

function get_Cξ_Cη(p::AbstractArray{<:Real,2}, t::AbstractArray{<:Integer,2}, C₀::AbstractArray{<:Real,3})
    # indices
	np = size(p, 1)
	nt = size(t, 1)

	# create global linear system using stamping method
    Cξ = Tuple{Int64,Int64,Float64}[]  
    Cη = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to Cξ from element k
        Cξᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = C₀[k, 2, j]*local_basis_func(C₀[k, :, i], ξ, η)
                Cξᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# calculate contribution to Cη from element k
        Cηᵏ = zeros(3, 3)
        for i=1:3
            for j=1:3
                func(ξ, η) = C₀[k, 3, j]*local_basis_func(C₀[k, :, i], ξ, η)
                Cηᵏ[i, j] = gaussian_quad2(func, p[t[k, :], :])
            end
        end

		# add to global system
		for i=1:3
			for j=1:3
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

function fem_evaluate(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real)
    fem_evaluate(v, ξ₀, η₀, m.p, m.t, m.C₀)
end
function fem_evaluate(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real, k::Integer)
    fem_evaluate(v, ξ₀, η₀, m.p, m.t, m.C₀, k)
end

function ∂ᵢ(u::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # find triangle p₀ is in
    k = get_tri(ξ₀, η₀, p, t)

    # evaluate there
    return ∂ᵢ(u, ξ₀, η₀, k, p, t, C₀; i)
end
function ∂ᵢ(u::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real, k::Integer, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # sum weighted combinations of c₂
    return dot(u[t[k, :]], C₀[k, i+1, :])
end
function ∂ᵢ(m::ModelSetup3DPG, u::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real; i::Integer)
    return ∂ᵢ(u, ξ₀, η₀, m.p, m.t, m.C₀; i)
end
function ∂ᵢ(m::ModelSetup3DPG, u::AbstractArray{<:Real,1}, ξ₀::Real, η₀::Real, k::Integer; i::Integer)
    return ∂ᵢ(u, ξ₀, η₀, k, m.p, m.t, m.C₀; i)
end

function ∂ξ(args...)
    return ∂ᵢ(args...; i=1)
end
function ∂η(args...)
    return ∂ᵢ(args...; i=2)
end

function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, ξ₀::Real, η₀::Real)
    # find triangle p₀ is in
    k = get_tri(ξ₀, η₀, m.p, m.t)

    # evaluate there
    return curl(m, u, ξ₀, η₀, k)
end
function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, ξ₀::Real, η₀::Real, k::Integer)
    return ∂ξ(m, u[2, :], ξ₀, η₀, k) - ∂η(m, u[1, :], ξ₀, η₀, k)
end