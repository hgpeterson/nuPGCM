struct ShapeFunctions{IN<:Integer, M<:AbstractMatrix, A<:AbstractArray}
    # order of polynomials defining shape functions
    order::IN

    # dimension of space
    dim::IN

    # number of nodes on reference element
    n::IN

    # coefficients matrix defining shape function polynomials
    C::M

    # coefficients matrices defining derivatives of shape function polynomials
    ∂C::A
end

"""
    s = ShapeFunctions(order, dim)

Construct shape functions of order `order` in dimension `dim`.
"""
function ShapeFunctions(order, dim)
    # get nodes of standard element
    p = reference_element_nodes(order, dim)
    n = size(p, 1)

    # perms[i] tells you the permutation at index i
    # if you want to know the index of a permutation, do `findall(x->x==1, indexin(perms, [perm]))[1]`
    perms = Vector{Tuple}([])
    for k=0:order
        k_perms = iter_permutations(k, dim) # get all permutations of `dim` integers s.t. their sum is k
        for i in eachindex(k_perms)
            push!(perms, k_perms[i])
        end
    end

    # compute shape function coefficients
	V = zeros(n, n)
    for row=1:n 
        for k=0:order
            k_perms = iter_permutations(k, dim) 
            for i in eachindex(k_perms)
                col = findall(x->x==1, indexin(perms, [k_perms[i]]))[1] # find index of k-permutation i
                V[row, col] = prod(p[row, :].^k_perms[i]) # ξˡηᵐζⁿ terms (note: julia says 0^0 is 1)
            end
        end
    end
 	C = inv(V)

    # compute shape function derivative coefficients
    ∂C = zeros(dim, n, n)
    for i=1:n
        perm = perms[i]
        for j=1:dim
            # take derivative in jth direction -> subtract 1 from jth term in permutation
            dperm = collect(perm) # convert to an array so we can modify an element
            if dperm[j] == 0
                # already constant in jth direction, derivative is zero
                continue
            end
            dperm[j] -= 1
            dperm = Tuple(dperm) # back to tuple to find in `perms`

            # find index of this permutation
            k = findall(x->x==1, indexin(perms, [dperm]))[1]

            # add perm[j] times original coefficient to ∂C
            ∂C[j, k, :] += perm[j]*C[i, :]
        end
    end

    return ShapeFunctions(order, dim, n, C, ∂C)
end

"""
    p = reference_element_nodes(order, dim)

The nodes of a reference element of order `order` in `dim` dimensions.
"""
function reference_element_nodes(order, dim)
    if dim == 1
        if order == 0
            return [0.0]
        elseif order == 1
            return [-1.0
                     1.0]
        elseif order == 2
            return [-1.0
                     1.0
                     0.0]
        elseif order == 3
            return [-1.0
                     1.0
                    -1/3
                     1/3]
        else
            error("Unsupported reference element order `$order` for dimension `$dim`.")
        end
    elseif dim == 2
        if order == 0
            return 1/3*[1.0  1.0]
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
            error("Unsupported reference element order `$order` for dimension `$dim`.")
        end
    elseif dim == 3
        if order == 0
            return 1/3*[1.0  1.0  1.0]
        elseif order == 1
            return [0.0  0.0  0.0
                    1.0  0.0  0.0
                    0.0  1.0  0.0
                    0.0  0.0  1.0]
        elseif order == 2
            return [0.0  0.0  0.0
                    1.0  0.0  0.0
                    0.0  1.0  0.0
                    0.0  0.0  1.0
                    0.5  0.0  0.0
                    0.5  0.5  0.0
                    0.0  0.5  0.0
                    0.0  0.0  0.5
                    0.5  0.0  0.5
                    0.0  0.5  0.5]
        else
            error("Unsupported reference element order `$order` for dimension `$dim`.")
        end
    else
        error("Unsupported reference element dimension `$dim`.")
    end
end


"""
    perms = iter_permutations(n, d)

Compute all permutations (i₁, i₂, ..., i_d) such that i₁ + i₂ + ... + i_d = n.
(Currently hard-coded for d = 1, 2, 3 only).
"""
function iter_permutations(n, d)
    perms = Vector{Tuple}([])
    if d == 1
        push!(perms, Tuple(n))
    elseif d == 2
        for i=0:n
            push!(perms, (n - i, i))
        end
    elseif d == 3
        for i=0:n
            k = n - i
            for j=0:k
                push!(perms, (k - j, j, i))
            end
        end
    end
    return perms
end

struct ShapeFunctionIntegrals{A2<:AbstractMatrix, A3<:AbstractArray, A4<:AbstractArray}
    # mass matrix of form ∫ φᵢ*φⱼ
    M::A2

    # C matrices of form ∫ ∂ξ₁(φᵢ)*φⱼ
    C::A3

    # Cᵀ matrices of form ∫ φᵢ*∂ξ₁(φⱼ)
    CT::A3

    # stiffness matrices of form ∫ ∂ξ₁(φᵢ)*∂ξ₂(φⱼ)
    K::A4
end

"""
    s = ShapeFunctionIntegrals(sf_trial, sf_test)

Compute and store integrals over the standard triangle [0 0; 1 0; 0 1] of the form
    ∫ ∂ₙφⱼ ∂ₘφᵢ dξ
where φᵢ and φⱼ are shape functions from the trial and test space, respectively.
"""
function ShapeFunctionIntegrals(sf_trial::ShapeFunctions, sf_test::ShapeFunctions) 
    # quadrature weights and points
    w, ξ = quad_weights_points(max(1, sf_trial.order + sf_test.order), 2)

    # for now
    dim = 2

    # mass
    M = compute_integral_matrix((ξ, i, j) -> φ(sf_trial, j, ξ)*φ(sf_test, i, ξ), w, ξ, sf_test.n, sf_trial.n)

    # C
    C = zeros(dim, sf_test.n, sf_trial.n)
    for k=1:dim
        C[k, :, :] = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, k, ξ)*φ(sf_test, i, ξ), w, ξ, sf_test.n, sf_trial.n)
    end

    CT = zeros(dim, sf_test.n, sf_trial.n)
    for k=1:dim
        CT[k, :, :] = compute_integral_matrix((ξ, i, j) -> φ(sf_trial, j, ξ)*∂φ(sf_test, i, k, ξ), w, ξ, sf_test.n, sf_trial.n)
    end
    
    # stiffness
    K = zeros(dim, dim, sf_test.n, sf_trial.n)
    for k=1:dim, l=1:dim
        K[k, l, :, :] = compute_integral_matrix((ξ, i, j) -> ∂φ(sf_trial, j, k, ξ)*∂φ(sf_test, i, l, ξ), w, ξ, sf_test.n, sf_trial.n)
    end

    return ShapeFunctionIntegrals(M, C, CT, K)
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

Evaluate shape function `i` at the point `ξ`.
"""
function φ(sf::ShapeFunctions, i, ξ)
    return eval_poly(sf.C[:, i], ξ, sf.order, sf.dim)
end

"""
    ∂φ(sf, i, j, ξ)

Evaluate `j`-derivative of shape function `i` at the point `ξ`.
"""
function ∂φ(sf::ShapeFunctions, i, j, ξ)
    return eval_poly(sf.∂C[j, :, i], ξ, sf.order, sf.dim)
end

"""
    f = eval_poly(c, ξ, n, d)

Evaluate `n` degree polynomial in `d` dimensions defined by coefficients `c` at point `ξ`.
"""
function eval_poly(c, ξ, n, d)
    f = 0 
    i = 1
    for k=0:n
        perms = iter_permutations(k, d)
        for j in eachindex(perms)
            f += c[i]*prod(ξ.^perms[j]) # (btw: julia says 0^0 is 1)
            i += 1
        end
    end
    return f
end

struct FEGrid{FM<:AbstractMatrix, IM<:AbstractMatrix, IV<:AbstractVector, IN<:Integer}
    # order of shape functions on this grid
    order::IN

    # dimension of space
    dim::IN

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

    # determine dimension
    dim = size(p, 2)

    # setup shape functions
    s = ShapeFunctions(order, dim)

    return FEGrid(order, dim, s, p, np, t, nt, nn, e, ne)
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

        # map to triangle data structure
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

        # not as easy to determine the indices for each triangle because the 1/3rd point for one triangle is
        # the 2/3rd point for another... this works but it is slow
        tnew = zeros(Int64, size(t, 1), n)
        ps = standard_element_nodes(order)
        tnew[:, 1:3] = t
        @showprogress "Triangulating 3rd-order mesh..." for k in axes(t, 1)
            for i=4:n-1
                p₀ = transform_from_std_tri(ps[i, :], pnew[t[k, :], :])
                idx = get_idx(pnew, p₀)
                tnew[k, i] = idx
            end
        end
        tnew[:, 10] = np2 .+ (1:size(t,1))'

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

struct Jacobians{V<:AbstractVector, A<:AbstractArray}
    # determinant of the Jacobian of each element
    detJ::V

    # Jacobian of each element
    J::A
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
    detJ = @. xξ*yη - xη*yξ
    J = zeros(g.nt, g.dim, g.dim)
    # for i=1:g.dim, j=1:g.dim
    #     # i = x, y, z
    #     # j = ξ, η, ζ
    #     x = g.p[:, mod1(i+1, g.dim)]
    #     ks = [4 - j, 1]
    #     k₁ = ks[i]
    #     k₂ = ks[mod1(i+1, 2)]
    #     J[:, i, j] = (x[g.t[:, k₁]] - x[g.t[:, k₂]])./detJ
    # end
    ξx = (y[g.t[:, 3]] - y[g.t[:, 1]])#./detJ
    ξy = (x[g.t[:, 1]] - x[g.t[:, 3]])#./detJ
    ηx = (y[g.t[:, 1]] - y[g.t[:, 2]])#./detJ
    ηy = (x[g.t[:, 2]] - x[g.t[:, 1]])#./detJ
    J[:, 1, 1] = ξx
    J[:, 2, 1] = ξy
    J[:, 1, 2] = ηx
    J[:, 2, 2] = ηy
    println(det(J[1, :, :]))
    println(detJ[1])
    return Jacobians(detJ, J)
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
function FEField(gfile::String, order::Integer, values)
    g = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)
    return FEField(order, values, g, g1)
end
function FEField(values, g::FEGrid, g1::FEGrid)
    return FEField(g.order, values, g, g1)
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
