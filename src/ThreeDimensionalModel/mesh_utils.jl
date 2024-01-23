function generate_wedge_cols(g_sfc1, g_sfc2; nσ=0, chebyshev=false)
    if nσ == 0
        # by default, compute nσ to roughly match surface mesh
        emap, edges, bndix = all_edges(g_sfc1.t)
        h = 1/size(edges, 1)*sum(norm(g_sfc1.p[edges[i, 1], :] - g_sfc1.p[edges[i, 2], :]) for i ∈ axes(edges, 1))
        nσ = Int64(round(1/h)) + 1
    end

    # σ coordinates for node grids
    if chebyshev
        σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
    else
        σ = collect(-1:1/(nσ-1):0)
    end

    # node points on second order grid
    p2 = hcat(repeat(g_sfc2.p, inner=(nσ, 1)), repeat(σ, g_sfc2.np))
    np2 = g_sfc2.np*nσ

    # node connectivities on second order grid
    t2 = zeros(Int64, g_sfc2.nt*(nσ - 1), 12)
    for k_sfc=1:g_sfc2.nt
        for j=1:nσ-1
            t2[get_k_w(k_sfc, nσ, j), 1:3]   = nσ*(g_sfc2.t[k_sfc, 1:3] .- 1) .+ j
            t2[get_k_w(k_sfc, nσ, j), 4:6]   = nσ*(g_sfc2.t[k_sfc, 1:3] .- 1) .+ j .+ 1
            t2[get_k_w(k_sfc, nσ, j), 7:9]   = nσ*(g_sfc2.t[k_sfc, 4:6] .- 1) .+ j
            t2[get_k_w(k_sfc, nσ, j), 10:12] = nσ*(g_sfc2.t[k_sfc, 4:6] .- 1) .+ j .+ 1
        end
    end

    # boundaries on second order grid
    e2 = Dict("sfc"=>collect(nσ:nσ:np2), 
              "bot"=>collect(1:nσ:np2-nσ+1),
              "coast"=>flatten(collect.(get_col_inds.(g_sfc2.e["bdy"], nσ))))

    # second order grid
    g2 = Grid(Wedge(order=2), p2, t2, e2)

    # first order grid
    np1 = g_sfc1.np*nσ
    e1 = Dict("sfc"=>collect(nσ:nσ:np1), 
              "bot"=>collect(1:nσ:np1-nσ+1),
              "coast"=>flatten(collect.(get_col_inds.(g_sfc1.e["bdy"], nσ))))
    g1 = Grid(Wedge(order=1), p2[1:np1, :], t2[:, 1:6], e1)

    return g1, g2, σ
end

#### index mappings

"""
    k_sfc = get_k_sfc(k_w, nσ) 

Returns index `k_sfc` of the surface triangle associated with the wedge of 
index `k_w` for a mesh with `nσ` vertical nodes.
"""
get_k_sfc(k_w, nσ) = div(k_w - 1, nσ - 1) + 1

"""
    i_sfc = get_i_sfc(i, nσ) 

Returns index `i_sfc` of the surface node associated with the `i`th node on
the 3D wedge mesh with `nσ` vertical nodes.
"""
get_i_sfc(i, nσ) = div(i - 1, nσ) + 1

"""
    k_w = get_k_w(k_sfc, nσ, j) 

Returns index `k_w` of the `j`th wedge that lies under the surface triangle of index
`k_sfc` for a mesh with `nσ` vertical nodes.
"""
get_k_w(k_sfc, nσ, j) = (k_sfc - 1)*(nσ - 1) + j

"""
    k_ws = get_k_ws(k_sfc, nσ) 

Returns indices `k_ws` of the wedges that lie under the surface triangle with index
`k_sfc` for a mesh with `nσ` vertical nodes.
"""
get_k_ws(k_sfc, nσ) = get_k_w(k_sfc, nσ, 1):get_k_w(k_sfc, nσ, nσ - 1)

"""
    inds = get_col_inds(i_sfc, nσ) 

Returns the indices `inds` for the nodes in the `i_sfc`th column of a mesh with `nσ` 
vertical nodes.
"""
get_col_inds(i_sfc, nσ) = get_i_bot(i_sfc, nσ):get_i_top(i_sfc, nσ)

"""
    i = get_i_bot(i_sfc, nσ) 

Returns the index `i` for the bottom node in the `i_sfc`th column of a mesh with `nσ` 
vertical nodes.
"""
get_i_bot(i_sfc, nσ) = (i_sfc - 1)*nσ + 1

"""
    i = get_i_top(i_sfc, nσ) 

Returns the index `i` for the top node in the `i_sfc`th column of a mesh with `nσ` 
vertical nodes.
"""
get_i_top(i_sfc, nσ) = i_sfc*nσ

"""
    Af = flatten(A)

Quick and dirty utility function to flatten a vector of vectors into a single vector.

# Examples

```
julia> A = [[1, 2], [3, 4]]
2-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4, 5]

julia> flatten(A)
5-element Vector{Int64}:
 1
 2
 3
 4
 5
```
"""
function flatten(A)
    n = sum(length(A[i]) for i ∈ eachindex(A))
    Af = zeros(typeof(A[1][1]), n)
    i = 1
    for j ∈ eachindex(A)
        Af[i:i+length(A[j])-1] = A[j]
        i += length(A[j])
    end
    return Af
end