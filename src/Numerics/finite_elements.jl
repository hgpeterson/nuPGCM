struct ShapeFunctions{IN<:Integer, M<:AbstractMatrix, A<:AbstractArray}
    # order of polynomials defining shape functions
    order::IN

    # number of nodes on standard element
    n::IN

    # coefficients matrix defining shape function polynomials
    C::M

    # array of coefficients matrices defining derivatives of shape function polynomials
    ∂C::A
end

"""
    s = ShapeFunctions(order)

Construct shape functions of order `order`.
"""
function ShapeFunctions(order)
    # get nodes of standard element
    p = standard_element_nodes(order)
    n = size(p, 1)

    # compute shape function coefficients
	V = zeros(n, n)
	for i=1:n
	    ξ = p[i, 1]
	    η = p[i, 2]
        if order == 0
            V[:, i] = [1]
        elseif order == 1
		    V[:, i] = [1, ξ, η]
        elseif order == 2
	        V[:, i] = [1, ξ, η, ξ^2, ξ*η, η^2]
        elseif order == 3
	        V[:, i] = [1, ξ, η, ξ^2, ξ*η, η^2, ξ^3, ξ^2*η, ξ*η^2, η^3]
        else
            error("Unsupported shape function order.")
        end
	end
	C = inv(V)

    # compute shape function derivative coefficients
    ∂C = zeros(2, n, n)
    for j=1:2
        if order == 0
            continue
        elseif order == 1
            ∂C[j, :, 1] += C[:, j+1]
        elseif order == 2
            ∂C[j, :, 1] += C[:, j+1]

            ∂C[j, :, j+1] += 2*C[:, 2j+2]
            ∂C[j, :, 4-j] += C[:, 5]
        elseif order == 3
            ∂C[j, :, 1]    +=   C[:, j+1]
            ∂C[j, :, j+1]  += 2*C[:, 2j+2]
            ∂C[j, :, 4-j]  +=   C[:, 5]
            ∂C[j, :, 2j+2] += 3*C[:, 3j+4]
            ∂C[j, :, 5]    += 2*C[:, j+7]
            ∂C[j, :, 8-2j] +=   C[:, 10-j]
        end
    end
    return ShapeFunctions(order, n, C, ∂C)
end

"""
    p = standard_element_nodes(order)

The nodes of a standard element of order `order`.
"""
function standard_element_nodes(order)
    if order == 0
        return 1/3*[1.0 1.0]
    elseif order == 1
        return [0.0  0.0
                1.0  0.0
                0.0  1.0]
    elseif order == 2
        return [0.0  0.0
                1.0  0.0
                0.0  1.0
                0.5  0.0
                0.5  0.5
                0.0  0.5]
    elseif order == 3
        return [0.0  0.0
                1.0  0.0
                0.0  1.0
                1/3  0.0
                2/3  0.0
                2/3  1/3
                1/3  2/3
                0.0  2/3
                0.0  1/3
                1/3  1/3]
    else
        error("Unsupported shape function order.")
    end
end

struct ShapeFunctionIntegrals{M<:AbstractMatrix}
    φφ::M
    φξφ::M
    φηφ::M
    φφξ::M
    φφη::M
    φξφξ::M
    φξφη::M
    φηφξ::M
    φηφη::M
end

"""
    s = ShapeFunctionIntegrals(sf_trial, sf_test)

Compute and store integrals over the standard triangle [0 0; 1 0; 0 1] of the form
    ∫ ∂ₙφⱼ ∂ₘφᵢ dξ
where φᵢ and φⱼ are shape functions from the trial and test space, respectively.
"""
function ShapeFunctionIntegrals(sf_trial::ShapeFunctions, sf_test::ShapeFunctions) 
    # quadrature weights and points
    w, ξ = quad_weights_points(max(1, sf_trial.order + sf_test.order))

    # mass
    φφ = compute_integral_matrix((ξ, i, j) -> φ(sf_trial, j, ξ)*φ(sf_test, i, ξ), w, ξ, sf_test.n, sf_trial.n)

    # C
    φξφ = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, 1, ξ)*φ(sf_test, i, ξ), w, ξ, sf_test.n, sf_trial.n)
    φηφ = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, 2, ξ)*φ(sf_test, i, ξ), w, ξ, sf_test.n, sf_trial.n)
    φφξ = compute_integral_matrix((ξ, i, j) -> φ(sf_trial, j, ξ)*∂φ(sf_test, i, 1, ξ), w, ξ, sf_test.n, sf_trial.n)
    φφη = compute_integral_matrix((ξ, i, j) -> φ(sf_trial, j, ξ)*∂φ(sf_test, i, 2, ξ), w, ξ, sf_test.n, sf_trial.n)

    # stiffness
    φξφξ = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, 1, ξ)*∂φ(sf_test, i, 1, ξ), w, ξ, sf_test.n, sf_trial.n)
    φξφη = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, 1, ξ)*∂φ(sf_test, i, 2, ξ), w, ξ, sf_test.n, sf_trial.n)
    φηφξ = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, 2, ξ)*∂φ(sf_test, i, 1, ξ), w, ξ, sf_test.n, sf_trial.n)
    φηφη = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, 2, ξ)*∂φ(sf_test, i, 2, ξ), w, ξ, sf_test.n, sf_trial.n)
    return ShapeFunctionIntegrals(φφ, φξφ, φηφ, φφξ, φφη, φξφξ, φξφη, φηφξ, φηφη)
end

"""
    M = compute_integral_matrix(f, w, ξ, n)

Compute integrals over the standard triangle of the form ∫ f(ξ, i, j) dξ 
for i = 1, .., n and j = 1, ..., m. Quadrature rule defined by weights `w` and integration 
points `ξ`.
"""
function compute_integral_matrix(f, w, ξ, n, m)
    M = zeros(n, m)
    for i=1:n
        for j=1:m
            M[i, j] = std_tri_quad(ξ -> f(ξ, i, j), w, ξ)
        end
    end
    return M
end

"""
    φ(sf, i, ξ)

Shape function `i` evaluated at the point `ξ`.
"""
function φ(sf::ShapeFunctions, i, ξ)
    return eval_poly(sf.C[i, :], ξ)
end

"""
    ∂φ(sf, i, j, ξ)

Derivative of shape function `i` in the `j` direction evaluated at the point `ξ`.
"""
function ∂φ(sf::ShapeFunctions, i, j, ξ)
    return eval_poly(sf.∂C[j, i, :], ξ)
end

"""
    f = eval_poly(c, ξ)

Evaluate polynomial defined by coefficients `c` at point `ξ`.
"""
function eval_poly(c, ξ)
    n = size(c, 1)
    if n == 1
        return c'*[1]
    elseif n == 3
        return c'*[1, ξ[1], ξ[2]]
    elseif n == 6
        return c'*[1, ξ[1], ξ[2], ξ[1]^2, ξ[1]*ξ[2], ξ[2]^2]
    elseif n == 10
        return c'*[1, ξ[1], ξ[2], ξ[1]^2, ξ[1]*ξ[2], ξ[2]^2, ξ[1]^3, ξ[1]^2*ξ[2], ξ[1]*ξ[2]^2, ξ[2]^3]
    else
        error("Unsupported polynomial order.")
    end
end

struct FEGrid{FM<:AbstractMatrix, IM<:AbstractMatrix, IV<:AbstractVector, IN<:Integer}
    # order of shape functions on this grid
    order::IN

    # shape functions on this grid
    s::ShapeFunctions

    # node positions
    p::FM # float matrix

    # number of nodes
    np::IN

    # node indices defining each element
    t::IM # integer matrix

    # number of elements
    nt::IN

    # number of nodes per element
    nn::IN

    # edge node indices
    e::IV # integer vector

    # number of edge nodes
    ne::IN
end

"""
    g = FEGrid(file_name, order)

Construct a FE grid of order `order` by loading points `p`, triangles `t`, and boundary nodes `e` from .h5 file.
"""
function FEGrid(file_name, order::IN) where IN <: Integer
    # setup shape functions
    s = ShapeFunctions(order)

    # read grid data
    file = h5open(file_name, "r")
    p = read(file, "p")
    t = read(file, "t")
    e = read(file, "e")
    e = e[:, 1]
    t = convert(Matrix{IN}, t)
    e = convert(Vector{IN}, e)
    if order == 0
        # only need centroids
        pp = zeros(size(t, 1), 2)
        ee = Vector{IN}()
        for k in axes(t, 1)
            pp[k, :] = 1/3*sum(p[t[k, :], :], dims=1)
            if t[k, 1] in e || t[k, 2] in e || t[k, 3] in e
                push!(ee, k)
            end
        end
        p = pp
        t = zeros(IN, size(t, 1), 1)
        t[:, 1] = 1:size(t, 1)
        e = ee
    elseif order > 1
        # add nodes for higher orders
        p, t, e = add_nodes(p, t, e, order)
    end
    np = size(p, 1)
    nt = size(t, 1)
    nn = size(t, 2)
    ne = size(e, 1)
    return FEGrid(order, s, p, np, t, nt, nn, e, ne)
end

"""
	p, t, e = add_nodes(p, t, e, order)

Add nodes to mesh for higher-order shape functions.
"""
function add_nodes(p, t, e, order)
    # get edges
    edges, boundary_indices, emap = all_edges(t)

    if order == 2
        # number of nodes per triangle
        n = 6

        # add midpoints
        np0 = size(p, 1)
        new_pts = 1/2*reshape(p[edges[:, 1], :] + p[edges[:, 2], :], (size(edges, 1), 2))
        pnew = [p; new_pts]

        # easy to map to triangle data structure
        tnew = zeros(Int64, size(t, 1), n)
        tnew[:, 1:3] = t
        tnew[:, 4:6] = np0 .+ emap

        # add points that were on the boundary to `e`
        enew = [e; np0 .+ boundary_indices]
    elseif order == 3
        # number of nodes per triangle
        n = 10

        # first add 1/3 points
        np0 = size(p, 1)
        new_pts = reshape(p[edges[:, 1], :] + 1/3*(p[edges[:, 2], :] - p[edges[:, 1], :]), (size(edges, 1), 2))
        pnew = [p; new_pts]
        np1 = size(pnew, 1)
        # then add 2/3 points
        new_pts = reshape(p[edges[:, 1], :] + 2/3*(p[edges[:, 2], :] - p[edges[:, 1], :]), (size(edges, 1), 2))
        pnew = [pnew; new_pts]
        np2 = size(pnew, 1)
        # finally add center points
        new_pts = reshape(1/3*(p[t[:, 1], :] + p[t[:, 2], :] + p[t[:, 3], :]), (size(t, 1), 2))
        pnew = [pnew; new_pts]

        # not as easy to determine the indices for each triangle... this works but it is slow
        tnew = zeros(Int64, size(t, 1), n)
        ps = standard_element_nodes(order)
        for k in axes(t, 1)
            for i in axes(ps, 1)
                p₀ = transform_from_std_tri(ps[i, :], pnew[t[k, :], :])
                idx = get_idx(pnew, p₀)
                tnew[k, i] = idx
            end
        end

        # add points that were on boundary to `e`
        enew = [e; np0 .+ boundary_indices]
        enew = [enew; np1 .+ boundary_indices]
    end

    # fig, ax, im = tplot(pnew, tnew)
    # # ax.plot(pnew[1:np0, 1], pnew[1:np0, 2], "o", ms=1)
    # # ax.plot(pnew[(np0+1):end, 1], pnew[(np0+1):end, 2], "o", ms=1)
    # # ax.plot(pnew[enew, 1], pnew[enew, 2], "wo", ms=0.5)
    # for k=[1, 6, 10]
    #     ax.plot(pnew[tnew[k, :], 1], pnew[tnew[k, :], 2], "o-", ms=1)
    # end
    # ax.axis("equal")
    # savefig("images/debug.png")
    # plt.close()

    return pnew, tnew, enew
end

"""
    i = get_idx(p, p₀)

Find the node index of point `p₀` in set of points `p`.
"""
function get_idx(p, p₀)
    Δp = @. (p[:, 1] - p₀[1])^2 + (p[:, 2] - p₀[2])^2
    return argmin(Δp)
end

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique `edges` (ne x 2 array) in the triangulation `t`.
Second output is indices to the boundary edges.
Third output `emap` (nt x 3 array) is a mapping from local triangle edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in triangle it.
"""
function all_edges(t)
    # find all edges
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)

    # remove duplicates
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]

    # compute local indices
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)

    # find boundary indices
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
    e = boundary_nodes(t)

Find all boundary nodes in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

struct Jacobians{V<:AbstractVector}
    J::V
    xξ::V
    xη::V
    yξ::V
    yη::V
    ξx::V
    ξy::V
    ηx::V
    ηy::V
end

"""
    J = Jacobians(g)

Compute Jacobian terms for transformations from standard triangle [0 0; 1 0; 0 1] to 
triangles on the grid. Given a triangle on the grid defined by the nodes [x1 y1; x2 y2; x3 y3], 
the transformation ξ ↦ x is defined by
    x = u1 + u2*ξ + u3*η
where
    u1 = [x1 y1]
    u2 = [x2-x1 y2-y1]
    u3 = [x3-x1 y3-y1].
The insverse transform x ↦ ξ is
    ξ = v1 + v2*x + v3*y
where
    v1 = 1/J*[y1*x3-x1*y3, x1*y2-y1*x2)
    v2 = 1/J*[y3-y1,       y1-y2)
    v3 = 1/J*[x1-x3,       x2-x1)
and J = ∂(x, y)/∂(ξ, η) is the jacobian.
"""
function Jacobians(g::FEGrid)
    # unpack coords
    x = g.p[:, 1]
    y = g.p[:, 2]

    # compute Jacobian terms for each triangle 
    xξ = x[g.t[:, 2]] - x[g.t[:, 1]]
    xη = x[g.t[:, 3]] - x[g.t[:, 1]]
    yξ = y[g.t[:, 2]] - y[g.t[:, 1]]
    yη = y[g.t[:, 3]] - y[g.t[:, 1]]
    J = @. xξ*yη - xη*yξ
    ξx = (y[g.t[:, 3]] - y[g.t[:, 1]])./J
    ξy = (x[g.t[:, 1]] - x[g.t[:, 3]])./J
    ηx = (y[g.t[:, 1]] - y[g.t[:, 2]])./J
    ηy = (x[g.t[:, 2]] - x[g.t[:, 1]])./J
    return Jacobians(J, xξ, xη, yξ, yη, ξx, ξy, ηx, ηy)
end
function Jacobians(gfile::String)
    # get order 1 FE grid
    g = FEGrid(gfile, 1)
    return Jacobians(g)
end

struct FEField{IN<:Integer,V<:AbstractVector}
    # order of polynomials defining shape functions
    order::IN

    # values of FE field on the nodes of the grid
    values::V

    # grid FE field exists on
    g::FEGrid

    # grid of order 1
    g1::FEGrid
end

"""
    u = FEField(gfile, order, values)

Construct FE field from grid saved at `gfile` of order `order` with node values `values`.
"""
function FEField(gfile, order, values)
    g = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)
    return FEField(order, values, g, g1)
end

"""
    l2 = L2norm(g, s, J, u)

Compute L2 norm, ‖u‖² ≡ ∫ u² dx, of finite element function `u`.
"""
function L2norm(g::FEGrid, s::ShapeFunctionIntegrals, J::Jacobians, u)
    L2 = 0
    for k=1:g.nt
        for i=1:g.nn
            for j=1:g.nn
                L2 += u[g.t[k, j]]*u[g.t[k, i]]*s.φφ[i, j]*abs(J.J[k])
            end
        end
    end
    return sqrt(L2)
end

"""
    h1 = H1norm(g, s, J, u)

Compute H1 norm, ‖u‖² ≡ ∫ u² + (∇u)⋅(∇u) dx, of finite element function `u`.
"""
function H1norm(g::FEGrid, s::ShapeFunctionIntegrals, J::Jacobians, u)
    H1 = 0
    for k=1:g.nt
        for i=1:g.nn
            for j=1:g.nn
                H1 += abs(J.J[k])*(u[g.t[k, j]]*u[g.t[k, i]])*(
                        s.φφ[i,j] +
                        s.φξφξ[i, j]*(J.ξx[k]^2       + J.ξy[k]^2) + 
                        s.φξφη[i, j]*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                        s.φηφξ[i, j]*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                        s.φηφη[i, j]*(J.ηx[k]^2       + J.ηy[k]^2)
                      )
            end
        end
    end
    return sqrt(H1)
end

"""
    t_dict = get_t_dict(p, t)

`t_dict[i]` returns a vector of integer indices for each triangle point `i` is in.
"""
function get_t_dict(p, t)
    t_dict = Dict{Int64, Vector{Int64}}()
    @showprogress "Creating mesh dictionary..." for i=1:size(p, 1)
        t_dict[i] = findall(vec(sum(t .== i, dims=2) .== 1))
    end
    return t_dict
end

"""
    x = transform_from_std_tri(ξ, p)

Transform point `ξ` defined on standard triangle [0 0; 1 0; 0 1] to x defined on 
triangle with vertices `p`.
"""
function transform_from_std_tri(ξ, p)
    return p[1, :] + (p[2, :] - p[1, :])*ξ[1] + (p[3, :] - p[1, :])*ξ[2]
end

"""
    ξ = transform_to_std_tri(x, p)

Transform point `x` defined on triangle with vertices `p` to standard triangle [0 0; 1 0; 0 1].
"""
function transform_to_std_tri(x, p)
    # jacobian
    J = (p[2, 1] - p[1, 1])*(p[3, 2] - p[1, 2]) - (p[3, 1] - p[1, 1])*(p[2, 2] - p[1, 2])
    v = 1/J * [p[1, 2]*p[3, 1] - p[1, 1]*p[3, 2]  p[1, 1]*p[2, 2] - p[1, 2]*p[2, 1]
               p[3, 2] - p[1, 2]                  p[1, 2] - p[2, 2]
               p[1, 1] - p[3, 1]                  p[2, 1] - p[1, 1]
              ]
    return v[1, :] + v[2, :]*x[1] + v[3, :]*x[2]
end

"""
    bool = pt_in_tri(x, p)

Determine if point `x` is in triangle with nodes `p`.
(See https://stackoverflow.com/a/2049593).
"""
function pt_in_tri(x, p)
    d₁ = pt_sign(x, p[1, :], p[2, :])
    d₂ = pt_sign(x, p[2, :], p[3, :])
    d₃ = pt_sign(x, p[3, :], p[1, :])

    has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
    has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

    return !(has_neg && has_pos)
end
function pt_sign(p₁, p₂, p₃)
    return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
end


"""
    k = get_tri(x, g)

Determine index `k` of triangle on grid `g` in which the point `x` lies.
"""
# function get_tri(x, p, t, t_dict)
#     closest_p = argmin((p[:, 1] .- x[1]).^2 + (p[:, 2] .- x[2]).^2)
#     for k=t_dict[closest_p] # just look at triangles closest_p is in
#         if pt_in_tri(x, p[t[k, 1], :], p[t[k, 2], :], p[t[k, 3], :])
#             return k
#         end
#     end
#     error("Cannot find triangle; p₀=($(x[1]), $(x[2])) is not inside mesh domain.")
# end
function get_tri(x, g::FEGrid)
    for k=1:g.nt 
        if pt_in_tri(x, g.p[g.t[k, :], :])
            return k
        end
    end
    error("Cannot find triangle; p₀=($(x[1]), $(x[2])) is not inside mesh domain.")
end

# function fem_evaluate(v::AbstractArray{<:Real,1}, ξ::Real, η::Real, p::AbstractArray{<:Real,2}, 
#                       t::AbstractArray{<:Integer,2}, t_dict::AbstractDict{IN, Vector{IN}}, 
#                       C₀::AbstractArray{<:Real,3}) where IN <: Integer
#     # find triangle (ξ, η) is in
#     k = get_tri(ξ, η, p, t, t_dict)
    
#     # evaluate there
#     return fem_evaluate(v, ξ, η, p, t, C₀, k)
# end
function fem_evaluate(u::FEField, x)
    try
        # find triangle x is in
        k = get_tri(x, u.g1)

        # evaluate there
        return fem_evaluate(u, x, k)
    catch
        # if triangle not found, return NaN
        println("p₀=($(x[1]), $(x[2])) outside mesh domain.")
        return NaN
    end
end
function fem_evaluate(u::FEField, x, k)
    # transform to standard triangle
    ξ = transform_to_std_tri(x, u.g1.p[u.g1.t[k, :], :])

    # sum weighted combinations of triangle k's basis functions at x
    return sum([u.values[u.g.t[k, i]]*φ(u.g.s, i, ξ) for i=1:u.g.nn])
end
