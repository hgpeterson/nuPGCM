module nuPGCM

using Gridap
using GridapGmsh
using Gmsh: gmsh
using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using CuthillMcKee
using JLD2
using LinearAlgebra
using SparseArrays
using Krylov
using PyPlot
using Printf

# unit vectors
x⃗ = VectorValue(1.0, 0.0, 0.0)
y⃗ = VectorValue(0.0, 1.0, 0.0)
z⃗ = VectorValue(0.0, 0.0, 1.0)

# gradients 
∂x(u) = x⃗⋅∇(u)
∂y(u) = y⃗⋅∇(u)
∂z(u) = z⃗⋅∇(u)

# directory where the output files will be saved
global out_dir = "."

"""
    set_out_dir!(dir)

Set the output directory where the results will be saved. The function creates 
the directory and two subdirectory:
- `dir`/images for plots, and 
- `dir`/data for data files.
"""
function set_out_dir!(dir)
    if out_dir != dir
        global out_dir = dir
        @info "Output directory set to '$dir'"
    end

    if !isdir(out_dir)
        @info "Creating directory '$out_dir'"
        mkdir(out_dir)
    end
    if !isdir("$out_dir/images")
        @info "Creating subdirectory '$out_dir/images'"
        mkdir("$out_dir/images")
    end
    if !isdir("$out_dir/data")
        @info "Creating subdirectory '$out_dir/data'"
        mkdir("$out_dir/data")
    end
end

# bool to turn on/off printing timings (default off)
const ENABLE_TIMING = Ref(false)

"""
    @ctime "description" expr

Conditionally run `@time` if `ENABLE_TIMING` is `true`.
"""
macro ctime(label, expr)
    quote
        if $ENABLE_TIMING[]
            @time $label $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

# include all the module code
include("architectures.jl")
include("utils.jl")
include("inputs.jl")
include("meshes.jl")
include("spaces.jl")
include("dofs.jl")
include("iterative_solvers.jl")
include("inversion.jl")
include("evolution.jl")
include("model.jl")
include("IO.jl")
include("plotting.jl")

export 
x⃗,
y⃗,
z⃗,
∂x,
∂y,
∂z,
out_dir,
set_out_dir!,
ENABLE_TIMING,
# architectures.jl
AbstractArchitecture,
CPU,
GPU,
on_architecture,
architecture,
# inputs.jl
Parameters,
SurfaceDirichletBC,
ConvectionParameterization,
EddyParameterization,
SurfaceFluxBC,
Forcings,
# meshes.jl
Mesh,
# spaces.jl
Spaces,
# dofs.jl
FEData,
# inversion.jl
InversionToolkit,
invert!,
# evolution.jl
EvolutionToolkit,
# model.jl
State,
Model,
set_b!,
set_state_from_file!,
run!,
# IO.jl
save_state,
save_vtk,
set_state_from_file!,
# plotting.jl
nan_eval,
plot_slice,
plot_profiles,
sim_plots,
plot_sparsity_pattern

end # module