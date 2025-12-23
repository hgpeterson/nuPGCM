using Gridap
using GridapGmsh
using BenchmarkTools

# n = 1000
# domain = (0, 2π, 0, 2π)
# partition = (n, n)
# model = CartesianDiscreteModel(domain, partition)
model = GmshDiscreteModel("meshes/bowl3D_0.05.msh")

order = 2
reffe = ReferenceFE(lagrangian, Float64, order)
D = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["bot", "sfc"])
C = TrialFESpace(D, [0, 0])

degree = order^2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# u = VectorValue(1, 1, 1)
u = interpolate_everywhere(x->sin(x[1]), C)
v = interpolate_everywhere(x->cos(x[2]), C)
w = interpolate_everywhere(x->sin(x[3]), C)
c = interpolate_everywhere(x->sin(x[1])*cos(x[2])*sin(x[3]), C)
∂x(u) = VectorValue(1, 0, 0)⋅∇(u)
∂y(u) = VectorValue(0, 1, 0)⋅∇(u)
∂z(u) = VectorValue(0, 0, 1)⋅∇(u)
# l(d) = ∫( u⋅∇(c)*d )*dΩ
# l(d) = ∫( c*d + u*∂x(c)*d + v*∂y(c)*d + w*∂z(c)*d)*dΩ
Δt = 0.1
α = 0.1
γ = 1
κ(x) = exp(-x[3])
l(d) = ∫( c*d - Δt*u*∂x(c)*d - Δt*v*∂y(c)*d - Δt*w*∂z(c)*d - α*γ*∂x(c)*∂x(d)*κ - α*γ*∂y(c)*∂y(d)*κ - α*∂z(c)*∂z(d)*κ )dΩ

# @benchmark assemble_vector($l, $D)

# a = SparseMatrixAssembler(D, D)
# rhs = zeros(C.space.nfree)
# @benchmark Gridap.FESpaces.assemble_vector!($l, $rhs, $a, $D)

function assemble_vector!(f, b, a, V)
  v = get_fe_basis(V)
  vecdata = Gridap.FESpaces.collect_cell_vector(V,f(v))
  fill!(b,zero(eltype(b)))
#   @time Gridap.FESpaces.numeric_loop_vector!(b,a,vecdata)
#   @time caches = save_caches(b,a,vecdata)
#   @time numeric_loop_vector!(b,a,vecdata,caches)
  numeric_loop_vector!(b,a,vecdata)
  Gridap.FESpaces.create_from_nz(b)
end

function numeric_loop_vector!(b,a,vecdata)
    strategy = Gridap.FESpaces.get_assembly_strategy(a)
    for (cellvec, _cellids) in zip(vecdata...)
        cellids = Gridap.FESpaces.map_cell_rows(strategy,_cellids)
        if length(cellvec) > 0
            rows_cache = array_cache(cellids)
            vals_cache = array_cache(cellvec)
            vals1 = getindex!(vals_cache,cellvec,1)
            rows1 = getindex!(rows_cache,cellids,1)
            add! = Gridap.FESpaces.AddEntriesMap(+)
            add_cache = Gridap.FESpaces.return_cache(add!,b,vals1,rows1)
            caches = add_cache, vals_cache, rows_cache
            @time Gridap.FESpaces._numeric_loop_vector!(b,caches,cellvec,cellids)
        end
    end
    b
end

@noinline function _numeric_loop_vector!(vec,caches,cell_vals,cell_rows)
  add_cache, vals_cache, rows_cache = caches
  @assert length(cell_vals) == length(cell_rows)
  add! = AddEntriesMap(+)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,vec,vals,rows)
  end
end

# @time assemble_vector(l, D)
a = SparseMatrixAssembler(D, D)
rhs = zeros(C.space.nfree)
assemble_vector!(l, rhs, a, D)
println()
@time assemble_vector!(l, rhs, a, D)
println()