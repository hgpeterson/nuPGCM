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
function ShapeFunctions(; order, dim)
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
    perms = iter_permutations(n, d)

Compute all permutations (i₁, i₂, ..., i_d) such that i₁ + i₂ + ... + i_d = n.
"""
function iter_permutations(n, d)
    if d == 1
        return [(n,)]
    end

    perms = Vector{Tuple}([])
    # loop over each possible first element of the perm, `i`
    for i=0:n
        # recursively call `iter_permutations` on the remaining elements
        for sub_perm ∈ iter_permutations(n-i, d-1)
            # use `...` to unpack `sub_perm` and add `i` to it
            push!(perms, (sub_perm..., i))
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

Compute and store integrals of the form 
    ∫ ∂ₙφⱼ ∂ₘφᵢ dξ
over reference element. Here φᵢ and φⱼ are shape functions from the test and trial space, respectively.
"""
function ShapeFunctionIntegrals(sf_trial::ShapeFunctions, sf_test::ShapeFunctions) 
    # dimension of space 
    dim = sf_trial.dim

    # quadrature weights and points
    w, ξ = quad_weights_points(deg=max(1, sf_trial.order + sf_test.order), dim=dim)

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

Compute integrals over reference element of the form ∫ f(ξ, i, j) dξ 
for i = 1, .., n and j = 1, ..., m. Quadrature rule defined by weights `w` and integration 
points `ξ`.
"""
function compute_integral_matrix(f, w, ξ, n, m)
    return [ref_el_quad(ξ -> f(ξ, i, j), w, ξ) for i=1:n, j=1:m]
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

# shortcuts
∂φ∂ξ(sf::ShapeFunctions, i, ξ) = ∂φ(sf, i, 1, ξ)
∂φ∂η(sf::ShapeFunctions, i, ξ) = ∂φ(sf, i, 2, ξ)
∂φ∂ζ(sf::ShapeFunctions, i, ξ) = ∂φ(sf, i, 3, ξ)

"""
    f = eval_poly(c, ξ, n, d)

Evaluate `n` degree polynomial in `d` dimensions defined by coefficients `c` at point `ξ`.
"""
function eval_poly(c, ξ, n, d)
    f = 0 
    i = 1
    # loop over each degree ≤ n
    for k=0:n
        # get all permutations of exponents
        perms = iter_permutations(k, d)

        # add each term of the form ξᵃηᵇζᶜ...
        for perm ∈ perms
            f += c[i]*prod(ξ.^perm) # (btw: julia says 0^0 is 1)
            i += 1
        end
    end
    return f
end