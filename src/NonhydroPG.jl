module NonhydroPG

using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot
using SparseArrays
using HDF5
using Printf

include("mesh_utils.jl")
include("plotting.jl")
include("IO.jl")

export 
chebyshev_nodes,
MyGrid,
get_p_t,
get_p_to_t,
nan_eval,
unpack_fefunction,
quick_plot,
plot_yslice,
plot_profiles,
plot_sparsity_pattern,
write_sparse_matrix,
read_sparse_matrix

end # module