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
                Mᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=7)
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
                # func(ξ, η) = C₀[k, 2, j]*shape_func(C₀[k, :, i], ξ, η)
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η)
                Cξᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=7)
            end
        end

		# calculate contribution to Cη from element k
        Cηᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                # func(ξ, η) = C₀[k, 3, j]*shape_func(C₀[k, :, i], ξ, η)
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η)
                Cηᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=7)
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

function fem_evaluate(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ::Real, η::Real)
    return fem_evaluate(v, ξ, η, m.p, m.t, m.C₀)
end
function fem_evaluate(m::ModelSetup3DPG, v::AbstractArray{<:Real,1}, ξ::Real, η::Real, k::Integer)
    return fem_evaluate(v, ξ, η, m.p, m.t, m.C₀, k)
end

function ∂ᵢ(u::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # find triangle p₀ is in
    k = get_tri(ξ, η, p, t)

    # evaluate there
    return ∂ᵢ(u, ξ, η, k, p, t, C₀; i)
end
function ∂ᵢ(u::AbstractArray{<:Real,1}, ξ::Real, η::Real, k::Integer, p::AbstractArray{<:Real,2}, 
            t::AbstractArray{<:Real,2}, C₀::AbstractArray{<:Real,3}; i::Integer)
    # sum weighted combinations of shape function derivatives
    # return dot(u[t[k, :]], C₀[k, i+1, :])
    ∂u = 0
    for j=1:3
        ∂u += u[t[k, j]]*shape_func(C₀[k, j, :], ξ, η; dξ=Int(i==1), dη=Int(i==2))
    end
    return ∂u
end
function ∂ᵢ(m::ModelSetup3DPG, u::AbstractArray{<:Real,1}, ξ::Real, η::Real; i::Integer)
    return ∂ᵢ(u, ξ, η, m.p, m.t, m.C₀; i)
end
function ∂ᵢ(m::ModelSetup3DPG, u::AbstractArray{<:Real,1}, ξ::Real, η::Real, k::Integer; i::Integer)
    return ∂ᵢ(u, ξ, η, k, m.p, m.t, m.C₀; i)
end

function ∂ξ(args...)
    return ∂ᵢ(args...; i=1)
end
function ∂η(args...)
    return ∂ᵢ(args...; i=2)
end

function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, ξ::Real, η::Real)
    # find triangle (ξ, η) is in
    k = get_tri(ξ, η, m.p, m.t)

    # evaluate there
    return curl(m, u, ξ, η, k)
end
function curl(m::ModelSetup3DPG, u::AbstractArray{<:Real,2}, ξ::Real, η::Real, k::Integer)
    return ∂ξ(m, u[2, :], ξ, η, k) - ∂η(m, u[1, :], ξ, η, k)
end
